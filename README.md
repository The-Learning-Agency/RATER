# RATER
Top-performing algorithms in the RATER (Robust Algorithms for Thorough Essay Rating) challenge to segment and label discourse in student argumentative essays

---

**Why do students write the way they do? And are they any good at it?**

Understanding the nuance of how students write remains a complex challenge – one that can be aided by deeper insight into how various writing components ultimately come together to form effective essays and other texts.

Granular writing feedback can help create better writers but teachers are often too overwhelmed to provide it as needed. So what can help? More knowledge about the different elements of student writing can aid in better development of customized AI, machine learning, and more effective, formative teacher feedback.

There are currently numerous automated writing feedback tools, but they all have limitations. Many often fail to identify the structural elements of argumentative writing, such as thesis statements or supporting claims, or they fail to evaluate the quality of these argumentative elements. Additionally, most available tools are proprietary or inaccessible to educators because of their cost. This problem is compounded for under-serviced schools which serve a disproportionate number of students of color and from low-income backgrounds. In short, the field of automated writing feedback is ripe for innovation that could help democratize education.

---

**The RATER Competition and Algorithms**

[The RATER (Robust Algorithms for Thorough Essay Rating) competition](https://the-learning-agency.com/robust-algorithms-for-thorough-essay-rating/overview/) sought to create an algorithm that can predict effective arguments and evaluate student writing overall so that students can have higher quality and more accessible automated writing tools. 

The challenge focused on creating more efficient versions of the algorithms developed in the original [Feedback Prize competition series](https://www.kaggle.com/competitions/feedback-prize-2021). The task was to develop an efficient “super-algorithm” that can:
Segment an essay into individual argumentative elements and classify each element according to its argumentative type (similar to the [“Feedback Prize - Evaluate Student Writing” Kaggle competition task](https://www.kaggle.com/competitions/feedback-prize-2021)), and
Evaluate the effectiveness of each discourse element (similar to the [“Feedback Prize - Predicting Effective Arguments” Kaggle competition task](https://www.kaggle.com/c/feedback-prize-effectiveness))

These models were developed using the [PERSUADE dataset](https://www.kaggle.com/datasets/julesking/tla-lab-persuade-dataset), a collection of 25,000 argumentative essays written by U.S. middle and high school students representing demographically diverse populations across socioeconomic and linguistic backgrounds.

This repository hosts the 2 top-performing RATER Algorithms:
* Baseline Algorithm - developed by [Andrija Milicevic](https://github.com/CroDoc)

    Performance: this model obtains a double-weighted F1 of 0.6473 on the custom metric (PDF description in Resources)
  
    Note: the Baseline Algorithm uses a different numeric labeling for the discourse types than the 1st Place Algorithm and the description in the [RATER competition data details](https://the-learning-agency.com/robust-algorithms-for-thorough-essay-rating/data/)
  
      0 -> Lead
      1 -> Position
      2 -> Claim
      3 -> Counterclaim
      4 -> Rebuttal
      5 -> Evidence
      6 -> Concluding Statement

* 1st Place Algorithm - developed by [Kossi Neroma](https://github.com/neroksi)

    Performance: this model obtains a double-weighted F1 of 0.6523 on the custom metric (PDF description in Resources)

---

The data used to train these models can be downloaded from Kaggle: [PERSUADE dataset](https://www.kaggle.com/datasets/julesking/tla-lab-persuade-dataset).

See the [RATER Competition Data page](https://the-learning-agency.com/robust-algorithms-for-thorough-essay-rating/data/) for additional data and model details.
