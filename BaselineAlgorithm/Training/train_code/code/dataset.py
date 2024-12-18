import torch
import re
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',
              'Evidence', 'Concluding Statement')

TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}

LABEL2EFFEC = ('Adequate', 'Effective', 'Ineffective')
EFFEC2LABEL = {t: l for l, t in enumerate(LABEL2EFFEC)}


class TextDataset(Dataset):
    def __init__(self,
                 text_df,
                 df,
                 tokenizer,
                 cfg,
                 validation=False,
                 ):
        
        self.text_df = text_df
        self.df = df
        self.tokenizer = tokenizer
        self.validation = validation

        self.max_length = cfg['max_length']

        self.samples = list(self.df.groupby('essay_id'))
        print(f'Loaded {len(self)} samples.')

        if self.validation:
            self.mask = 0
        else:
            self.mask = cfg['mask']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]

        text = self.text_df.loc[text_id, 'full_text']


        if self.validation:
            tokens = self.tokenizer(
                text,
                add_special_tokens = True,
                return_offsets_mapping=True
                )
        else:
            tokens = self.tokenizer(
                text,
                add_special_tokens = True,
                max_length = self.max_length,
                truncation=True,
                return_offsets_mapping=True
                )
        
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        if not (ious > 0).any(-1).all():
            print(text_id, len(text_id), len(tokens['input_ids']))
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        # word slices of ground truth spans
        gt_spans = []
        for _, row in df.iterrows():
            winds = row['predictionstring'].split()
            start = int(winds[0])
            end = int(winds[-1])
            span_label = TYPE2LABEL[row['discourse_type']]
            span_effectiveness = EFFEC2LABEL[row['discourse_effectiveness']]
            gt_spans.append([start, end + 1, span_label, span_effectiveness])
        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if self.mask > 0:
            all_inds = np.arange(1, len(input_ids) - 1)
            mask_ratio = random.random() * self.mask
            n_mask = max(int(len(all_inds) * mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(text=text,
                    text_id=text_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)


class CustomCollator(object):
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)

class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_df = None,
        train_df = None,
        valid_df = None,
        predict_df = None,
        tokenizer = None,
        cfg = None,
    ):
        super().__init__()
        self.text_df = text_df
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df

        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage):

        if stage in ['fit', 'validate']:
            self.train_dataset = TextDataset(self.text_df, self.train_df, self.tokenizer, self.cfg, validation=False)
            self.valid_dataset = TextDataset(self.text_df, self.valid_df, self.tokenizer, self.cfg, validation=True)
        elif stage == 'predict':
            self.predict_dataset = TextDataset(self.text_df, self.predict_df, self.tokenizer, self.cfg, validation=True)
        else:
            raise Exception()

    def train_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer.pad_token_id)
        return DataLoader(self.train_dataset, **self.cfg["train_loader"], collate_fn=custom_collator)

    def val_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer.pad_token_id)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)

    def predict_dataloader(self):
        custom_collator = CustomCollator(self.tokenizer.pad_token_id)
        return DataLoader(self.predict_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)
