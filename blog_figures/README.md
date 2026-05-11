# Blog Figures
This directory contains all of the plots and graphs displayed on our blog which were created using functions inside [model](../model/logo-classifier.ipynb) in cell
12. There are a total of 7 visualization-related functions used in our pipeline.
<br >
<br />

> [!NOTE]
> `create_accuracy_loss_plot(H)` relies on the history object, which is returned by training the model, being in memory and cannot work with just pre-saved weights of a model.

```
blog_figures/
│   ├── accuracy_loss_block3-top.png                     # create_accuracy_loss_plot(H)                                        
│   ├── confusion_matrix_block3-top.png                  # create_confusion_matrix(model, test_data, classes)
│   ├── f1_per_class_block7-top.png                      # create_f1_visual(top_k_df)
│   ├── heatmap_seattleymca_block3-top.png               # create_heatmap(img_array, model)
│   ├── heatmap_uinet_block3-top.png                     # create_heatmap(img_array, model)
│   └── top_k_results_block3-top.png                     # display_top_k(results_df, k=5)
```

`create_accuracy_loss_plot(H)` takes the training history object and plots train vs. validation accuracy and loss side-by-side across epochs to see how the model converged.

`create_confusion_matrix(model, test_data, classes)` runs the model on the test set, takes the argmax of the predicted probabilities, and plots the resulting confusion matrix against the true labels to show where the model is misclassifying between sectors.

`create_f1_visual(top_k_df)` computes per-class F1 scores from the top-1 predictions, sorts them ascending, and plots them as a horizontal bar chart with the macro F1 drawn as a reference line to highlight which sectors are dragging the average down ([source1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [source2](https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/)).

`create_heatmap(img_array, model)` implements Grad-CAM by computing the gradient of the predicted class score with respect to the last conv layer's output, then weights the activation maps by the pooled gradients to produce a class-discriminative heatmap ([source1](https://keras.io/examples/vision/grad_cam/), [source2](https://arxiv.org/abs/1610.02391)).

`display_top_k(results_df, k=5)` groups top-1 and top-k correctness per class into a horizontal bar chart and plots a side-by-side histogram of top-1 probabilities split by correct vs. wrong predictions to show how well-calibrated the model's confidence is ([source1](https://scikit-learn.org/stable/modules/calibration.html), [source2](https://arxiv.org/abs/1706.04599))
