# Blog Figures
This directory contains all of the plots and graphs displayed on our blog which were created using functions inside [model](../model/logo-classifier.ipynb) in cell
12. There are a total of 7 visualization-related functions used in our pipeline.
<br />

> [!NOTE]
> `create_accuracy_loss_plot(H)` relies on the history object, which is returned by training the model, being in memory and cannot work with just pre-saved weights of a model.  

```
blob_figures/                                            
│   ├── confusion_matrix_block3-top.png                  # create_confusion_matrix(model, test_data, classes)
│   ├── f1_per_class_block7-top.png                      # create_f1_visual(top_k_df)
│   ├── heatmap_seattleymca_block3-top.png               # create_heatmap(img_array, model)
│   ├── heatmap_uinet_block3-top.png                     # create_heatmap(img_array, model)
│   └── top_k_results_block3-top.png                     # display_top_k(results_df, k=5)
```
