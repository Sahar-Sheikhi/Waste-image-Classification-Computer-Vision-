# Waste-image-Classification-Computer-Vision-
Computer Vision Course Final Project


Waste Classification for Recyclability Status
A Comparison Between Vision Transformer (ViT), VGG16, Xception, ResNeXt, EfficientNet, and MLP


 
Sahar 	Sheikhi
sahar.sheikhi@studenti.unipd.it


 
ABSTRACT
Efficient and accurate classification of waste into recyclability categories is essential for optimizing recycling processes. This project aims to classify images of single pieces of waste into six categories: glass, paper, metal, plastic, cardboard, and trash. We are utilizing the dataset made publicly available by Yang et al. on GitHub, which contains approximately 400-500 images per class. We compared the performance of eight models, including five advanced convolutional neural network (CNN) models: Xception, VGG16, EfficientNet, and ResNeXt. Additionally, we evaluated two MLP models: a two-layer MLP (64-32), a three-layer MLP (256-128-64) and Vision Transformer (ViT) which introduces a novel approach using self-attention mechanisms to capture global context. These models were selected based on their state-of-the-art performance in image classification tasks. Our experiments aim to provide a comprehensive comparison of these models' performance on the waste classification task. We also compare implementations using transfer learning versus custom implementation to offer insights into the most effective architectures for practical applications in waste management.


1. Introduction

      Recycling plays a crucial role in building a sustainable society. Currently, recycling facilities rely on manual sorting and extensive filtering systems to separate materials, a process prone to human error and inefficiency. Moreover, consumers often struggle to correctly dispose of the diverse range of packaging materials available.Our project seeks to introduce an automated solution for classifying trash. This innovation has the potential to enhance the efficiency of processing plants and reduce waste by minimizing inaccuracies associated with manual sorting. Such advancements not only promise environmental benefits but also offer economic advantages Our project focuses on developing a method to classify single-object images into six distinct garbage categories using computer vision. This technology can mimic the material sorting process in recycling plants or assist consumers in identifying materials through image recognition alone. By employing various models including CNNs and MLPs, we aim to achieve accurate classification solely based on visual information, paving the way for more effective recycling practices.



The rest of this document is structured as follows: Section 2 discusses the similarities and differences between previous works and ours. In Section 3, we explain the dataset used, how it was created, and the preprocessing and data augmentation techniques employed. Section 4 outlines our approach to solving the problem. Section 5 details the implementation details and results and analysis. Section 6 explains experiment. Section 7 discusses the results and analysis of the implementation. Finally, in section 8 there are the conclusion and the future works.


2. Related Works

In earlier research, numerous image classification research projects using support vector machines and neural networks have been conducted. However, none have specifically addressed trash classification. One notable project is "Classification of Trash for Recyclability Status" by Mindy Yang and Gary Thung from Stanford University. Their project focuses on using AlexNet, a well-known CNN architecture, for trash classification. AlexNet, which won the 2012 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC), is relatively simple and known to perform well. Yang and Thung's project includes a dataset of 2,400 images across six classes, hand-collected and augmented to improve generalization [1].
A project from the 2016 TechCrunch Disrupt Hackathon created "Auto-Trash," an auto-sorting trash that can distinguish between compost and recycling using a Raspberry Pi powered module and camera [2]. The project was built using Google’s TensorFlow and included hardware components. It is notable that Auto-Trash only classifies whether something is compost or recycling, which is simpler than having five or six classes.
Another project involved a smartphone application designed to coarsely segment a pile of garbage in an image. The goal of the application was to allow citizens to track and report garbage in their neighborhoods. The dataset used was obtained through Bing Image Search, and patches were extracted from the images to train the network. A pre-trained AlexNet model was utilized, obtaining a mean accuracy of 87.69%, demonstrating the advantage of using a pre-trained model to improve generalization.
Other recycling-based classification problems used physical features of objects. In 1999, a project from Lulea University of Technology focused on recycling metal scraps using a mechanical shape identifier [3]. Chemical and mechanical methods such as probing were used to identify the chemical contents and current separation. This mechanical approach provides interesting advancement strategies for our project.
Another image-based classification of materials was performed on the Flickr Materials Database [4]. Features such as SIFT, color, micro texture, and outline shape were used in a Bayesian computational framework. This project aligns with the goal of classifying images based on material classes. However, the dataset used by Yang and Thung differs significantly; it includes hand-collected images of recycled objects, ensuring a more realistic representation with various states of wear, logos, and deformations. Another trash image classification performed is "Classification of Trash for Recyclability Status" [6] which takes images of a single piece of recycling or garbage and classifies it into six classes consisting of glass, paper, metal, plastic, cardboard, and trash. The models used are support vector machines (SVM) with scale-invariant feature transform (SIFT) features and a convolutional neural network (CNN). Our project is similar to them [6] implemented eight models including five CNN models: Xception, ResNet50, VGG16, EfficientNet, ReNeXt, one two-layer MLP 64-32, one three-layer MLP 256-128-64 and Vision Transformer (ViT) which introduces a novel approach using self-attention mechanisms to capture global context. It also performs a comparison between implementation of the models using pretrained models (Transfer Learning) and implementation without using pretrained models.

3 Dataset

3.1 Overview

         The dataset used in this project was carefully put together by Mindy Yang and Gary Thung from Stanford University, as described in their research on "Classification of Trash for Recyclability Status." They collected the data themselves because there weren't any existing datasets available for garbage materials. They initially considered using the Flickr Material Database and Google Images but found that these images didn't accurately represent recycled materials as they appear in real recycling plants. So, they gathered their own set of images, totaling about 2,400 photos across six categories: glass, paper, metal, plastic, cardboard, and trash. Each category, except "trash," had between 400 to 500 images. They took these photos against a white background in various locations like Stanford and their homes. To make the dataset more useful, they applied techniques like rotating, adjusting brightness, moving, resizing, distorting, and adjusting the average and normalizing. For this project, we utilized the GitHub repository provided by the authors of "Classification of Trash for Recyclability Status" from Stanford University. The repository, accessible at [7], contains the dataset used in their research. 


3.2 Preprocessing and Data augmentation

       The dataset collected by Mindy Yang and Gary Thung was subjected to several preprocessing and data augmentation techniques to enhance the generalization capability of the models. This dataset has been pre-processed and resized, making it convenient for our study on trash classification. By leveraging this resource, we were able to access a standardized dataset that aligns with the parameters set forth by Yang and Thung, facilitating consistency and comparability in our experiments.
Total number of images is 2527 including 137 trash ,501 glass, 594 paper images, 403 cardboard, 482 plastic, 410 metal images
The images were taken against a white background, and augmentation techniques such as rotating, adjusting brightness, moving, resizing, distorting, and normalizing were applied to increase the diversity of the dataset. This preprocessing ensures the models can learn robust features that generalize well to unseen data.

4. Method

In our work, we implemented eight models including five CNN models: Xception, ResNet50, VGG16, EfficientNet, ReNeXt, one two-layer MLP 64-32, one three-layer MLP 256-128-64. Every network takes as input a 224×224 RGB image and outputs the scores for each of the 6 classes of trash present in our dataset (paper, metal, cupboard, plastic, glass, trash). All models use a softmax classifier in final layer that provides classification probabilities for each class. Table [1] is an overview of models considered in this project.
 

4.1 Xception

 The Xception model deployed in this project addresses the challenge of trash image classification by leveraging its deep architecture and efficient feature extraction capabilities. With an input size of 224x224 pixels. Xception stands out due to its unique inception-style architecture, which employs depthwise separable convolutions to enhance feature learning while minimizing computational costs. This approach allows the model to effectively capture both local and global features critical for distinguishing between different types of waste items. By utilizing pre-trained weights from ImageNet, the model initializes with learned features that are further fine-tuned for the specific task. The addition of global average pooling layers condenses the extracted features, followed by dense layers for final classification using a softmax activation function. This design optimizes both computational efficiency and accuracy, making Xception well-suited for trash classification tasks where robust feature extraction and nuanced understanding of image content are paramount. 

Model	Architecture	Features
Xception	A CNN with depthwise separable convolutions, making it efficient and powerful.	Uses separable convolutions for efficiency.
ResNet50	A CNN with 50 layers, using shortcut connections to make training easier.	
Uses shortcuts to help with training deeper networks

VGG16	A CNN with 16 layers, using small (3x3) convolution filters, followed by three fully connected layers.	Simple, deep network with many parameters.
EfficientNet	A CNN that scales the network depth, width, and resolution systematically for better performance.	
Balances network size for efficiency and performance
ResNeXt	A CNN that combines several paths with a split-transform-merge strategy, increasing the number of paths.	
Uses multiple paths to increase model capacity.
MLP 64-32	A Multi-Layer Perceptron with two layers, the first with 64 neurons and the second with 32 neurons.	Simple, fully connected layers.
MLP 256-128-64	A Multi-Layer Perceptron with three layers, having 256, 128, and 64 neurons respectively.	More complex MLP with more layers and neurons.
ViT (Vision Transformer)	Uses a Transformer architecture, treating image patches as sequences and applying attention mechanisms.	Uses attention instead of convolutions for images.
Table 1. Overview of the models
 
4.2 EfficientNet

The EfficientNet model in this project leveraging pre-trained weights from ImageNet and fine-tuning on the specific task. With an initial base model setup using EfficientNetB0, which includes layers for feature extraction from images resized to 224x224 pixels, the model enhances representation through global average pooling and two dense layers (1024 units, ReLU activation, and softmax output). This approach optimizes performance by focusing on efficient feature extraction and leveraging transfer learning benefits. The choice of EfficientNet is justified by its balance between model size and accuracy, making it suitable for tasks like trash classification where computational efficiency and effective feature extraction are paramount.

  
Figure 1 . Example of an architecture EfficientNetB0  based on [8]

4.3 Vision Transformer (ViT)

The Vision Transformer (ViT) model implemented in this paper addresses the challenge of trash image classification by leveraging a patch-based strategy, dividing 224x224 pixel images into 16x16 patches. This approach is apt for the task as it allows the model to effectively capture both local and global features essential for distinguishing between different types of trash items. The ViT architecture's use of transformer layers, including multi-head self-attention and MLPs, facilitates the extraction of intricate patterns and relationships within and across patches, surpassing the limitations of traditional CNN-based methods. This ensures robust performance in image classification tasks where understanding both detailed local features and broader context is crucial.

 
Figure 2. Example of an architecture of the ViT, based on [9].

4.4 ResneXt

The ResNeXt model utilizes a ResNet-50 base architecture pretrained on ImageNet for trash image classification. The model's convolutional base is frozen to leverage its learned features, enhancing efficiency and training speed. Global average pooling is applied to condense the extracted features, followed by a softmax layer for classification. This streamlined architecture efficiently captures intricate image details and spatial hierarchies crucial for accurate trash item categorization. 
ResNeXt's robustness and effectiveness in feature learning make it a suitable choice for tasks requiring high-dimensional image analysis and classification.





 
Figure 3. Example of an architecture of Resnet50 based on [10]


4.5 VGG16

         The VGG16 model architecture used for waste image classification in this project consists of a pre-trained VGG16 base model with 13 convolutional layers and 3 fully connected layers, totaling 16 weight layers. This base model, which excludes the top classification layers, processes input images resized to 224x224 pixels. The output of the VGG16 base is fed into a GlobalAveragePooling2D layer to condense the feature maps, followed by a dense layer with 1024 units and ReLU activation to enhance feature representation. The final output layer is a dense layer with softmax activation corresponding to the number of waste classes, providing the classification probabilities. 

 

 
Figure 4. an example of architecture of VGG16 based on [11]

5.  Experiments

5.1 Implementation Details 

       To address class imbalance in our dataset, we balanced each class to have 200 samples using oversampling for classes with fewer than 200 samples by duplicating existing samples, and undersampling for classes with more than 200 samples by randomly selecting 200 samples. The balancing was helpful although we used a smaller dataset, we did not have any problem for training as in "Classification of Trash for Recyclability Status" [6] because of training trouble they had to remove the trash class since it has less images. We also normalized the dataset by scaling pixel values to a range of [0, 1], saving the normalized images back to their original paths. Additionally, we split the dataset into training, validation, and test sets, comprising 1000, 100, and 100 images respectively, and saved each subset to separate CSV files for future use.
 
We selected Adam because experiences showed that it works better than other optimizers. We have observed that 0.0001 is the best value that allows convergence faster to the best results, therefore we use this value for all models.
Data augmentation (which applies random transformations such as rotation, width/height shifts, horizontal flips, zoom, brightness adjustments) is used for models. Each model employs a batch size of 32 and processes images resized to 224x224 pixels for input and the number of epochs for all experiments set to 10. We use Adam optimizer with a learning rate of 0.0001 and the sparse categorical cross-entropy loss function, suitable for multi-class classification tasks.

 5.2 Results

      In figure [5] we plot the result of models ordered by the best test accuracy. Xception has the highest test accuracy while MLP  has the lowest.
 
Figure 5. Comparison of test accuracy for all models implemeneted by pretrained models

Table [3] shows the performance of different models, both with and without transfer learning, was evaluated using a dataset containing an equal portion of images from six waste classes. The results show a notable difference between the performance of models with and without transfer learning. Without transfer learning, the highest test accuracy achieved was 0.57 by the Xception model, while EfficientNet had the lowest at 0.27. This disparity can be attributed to the complexity and depth of the respective architectures. Xception, known for its depthwise separable convolutions, allows for more intricate feature extraction and representation compared to EfficientNet, which, although a powerful model, may struggle with capturing fine details and variations in the waste images without the benefit of pre-trained weights. However, when employing transfer learning, the accuracy significantly improved across all models. Xception, EfficientNet, and Vision Transformer (ViT) achieved test accuracies of 0.93, 0.89, and 0.86 respectively, demonstrating their superior capability in leveraging pre-trained weights for this task. VGG16 and ResNet50 also showed marked improvement with transfer learning, achieving test accuracies of 0.83 and 0.86 respectively, compared to 0.56 and 0.42 without it. Interestingly, the MLP model with the 256-128-64 architecture performed poorly in comparison, with test accuracies of 0.35 without transfer learning and 0.33 with transfer learning, indicating their limitations in handling image classification tasks without convolutional features. These results underscore the effectiveness of transfer learning in improving model performance for waste classification, suggesting that advanced CNN architectures like Xception and EfficientNet are highly suitable for practical applications in waste classification.



Model	without Transfer Learning	Transfer Learning
	Test Accuracy	Train
Accuracy	Test Accuracy	Train
Accuracy
Xception	0.57	0.61	0.93	0.96
Resnet50	0.42	0.49	0.86	0.90
EfficientNet	0.27	0.28	0.89	0.94
VGG16	0.56	0.58	0.83	0.96
ViT	0.51	0.58	0.86	0.92
MLP 256-128-64	0.35	0.41	0.33	0.53
Table 2. Test and train accuracy result for all models

Both implementations utilize similar datasets, preprocessing, label encoding, and evaluation methods, with the main difference being the model architecture: Implementation 1 leverages the pre-trained models for transfer learning, while Implementation 2 builds a custom model from scratch. This highlights how transfer learning can potentially offer faster and more accurate training, whereas custom models provide more flexibility and control over design. Both are implemented in TensorFlow, chosen for its ease of use, flexibility, access to pre-trained models, scalability, strong community support, and comprehensive ecosystem, making it ideal for developing and deploying machine learning models, including waste classification tasks. models are implemented on a local machine using Google Colab with a T4 GPU runtime.

Figure [6] shows the test and validation accuracy for the EfficientNet. As you see Adam optimizer works better than SGD for this task. Adam shows smoother convergence or less overfitting indicators compared to SGD figure [7], which uses a fixed learning rate for all parameters.



 
Figure 6. Training and validation accuracy for EfficientNet using Adam
 
Figure 7. Training and validation accuracy for EfficientNet using SGD

We used the data augmentation for all models, but we experienced that the test accuracy did not improve. 
Our model already has a high validation accuracy (94%) and test accuracy (89%). Since the initial performance is already high, improvements from augmentation might be marginal or not evident without more fine-tuning. Figure [8] is random representation of the dataset which shows that the dataset already contains images that were augmented during its creation, applying additional data augmentation during training might not lead to significant improvements in performance.

 
Figure 8. A random inspection of the images from dataset

Classification Report for Validation Data for EfficientNet:
Class        Precision    Recall       F1-score     Support     
------------------------------------------------------------
cardboard    1.0000       1.0000       1.0000       20          
glass        0.9286       0.8667       0.8966       15          
metal        1.0000       0.7895       0.8824       19          
paper        0.8750       1.0000       0.9333       14          
plastic      0.7692       1.0000       0.8696       10          
trash        1.0000       1.0000       1.0000       22          
------------------------------------------------------------
Validation Accuracy: 0.9400      
 
Classification Report for Test Data for EfficientNet:
Class        Precision    Recall       F1-score     Support     
------------------------------------------------------------
cardboard    1.0000       0.9091       0.9524       22          
glass        0.8462       0.8462       0.8462       13          
metal        1.0000       0.8500       0.9189       20          
paper        0.8000       1.0000       0.8889       12          
plastic      0.8235       0.8235       0.8235       17          
trash        0.8333       0.9375       0.8824       16          
------------------------------------------------------------
Test Accuracy: 0.8900      

Validation Confusion matrix figure [9] provides insight into our model's performance on the validation set. Each row corresponds to the true label of the waste material, while each column represents the predicted label. Test confusion matrix figure[10] shows the performance of our model on the test set, which serves as an indicator of how well our model generalizes to new, unseen data. Cardboard, Metal, Paper, and Trash classes are generally well-classified.
Glass items are frequently misclassified, especially as plastic. Plastic items are also well-classified, but there is some confusion with metal and paper.
 
Figure 9. Test confusion Matrix

 
Figure 10. Validation confusion matrix

7. Conclusion and Future work

In this study, we compared the performance of Vision Transformer (ViT), VGG16, Xception, ResNeXt, EfficientNet, and MLP on the task of waste image classification. We experimented with different architecture of each model combined with Adam optimizer.
Xception has the highest test accuracy, followed by Resnet50 and Vgg16.  MLP has the lowest performance for the waste classification task. 
Overall, Xception is the best performing model in terms of both validation and test accuracy. This suggests that Xception is the most accurate model for the given task.







      Future work could explore the integration of additional data augmentation techniques and the use of ensemble methods to further boost performance. Additionally, applying these models to more diverse and larger datasets could provide further insights into their generalization capabilities. Investigating the application of these models in real-world recycling systems could also be a valuable direction for future research











