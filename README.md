# AudioExplorers2023


# Transformer: 
2.6 Mb
Epoch 23/30
992/992 [==============================] - 106s 107ms/step - loss: 0.6344 - accuracy: 0.7685 - val_loss: 0.7400 - val_accuracy: 0.7357

# CNN
5 Mb, Params:  422.021,
Epoch 10/10 1323/1323 [==============================] - 115s 87ms/step - loss: 0.3506 - accuracy: 0.8737 - val_loss: 0.6444 - val_accuracy: 0.8110

# MobileNet without augmentation (overfitting):
33Mb, Params:  3.360.709, 
End of epoch 10: Accuracy: 0.7973151824541501, F1 Score: 0.7917057833068074, Precision: 0.7951711035668845, Recall: 0.7973151824541501
1323/1323 [==============================] - 235s 177ms/step - loss: 0.3319 - accuracy: 0.8755 - val_loss: 0.6121 - val_accuracy: 0.7973


# MobileNet with augmentation:
33Mb, Params:  3.360.709,
End of epoch 10: Accuracy: 0.7672527888069578, F1 Score: 0.762308391182108, Precision: 0.7687206539521811, Recall: 0.7672527888069578
1322/1322 [==============================] - 308s 233ms/step - loss: 0.6357 - accuracy: 0.7656 - val_loss: 0.6528 - val_accuracy: 0.7673

