**Title**: Same-different problems strain convolutional neural networks  
**Abstract**: The robust and efficient recognition of visual relations in images is a hallmark of biological vision. We argue that, despite recent progress in visual recognition, modern machine vision algorithms are severely limited in their ability to learn visual relations. Through controlled experiments, we demonstrate that visual-relation problems strain convolutional neural networks (CNNs). The networks eventually break altogether when rote memorization becomes impossible, as when intra-class variability exceeds network capacity. Motivated by the comparable success of biological vision, we argue that feedback mechanisms including attention and perceptual grouping may be the key computational components underlying abstract visual reasoning.  
**Link**: https://arxiv.org/abs/1802.03390 (edited)  


Rajat Kanti Bhattacharjee [22 hours ago]  
**Title**:Neural Turing Machine, 2014  
**Abstract** : We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent. Preliminary results demonstrate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples  
**Link**: https://arxiv.org/abs/1410.5401 (edited)  


z0k [22 hours ago]  
**Title**: How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)  
**Abstract**: Batch Normalization (BatchNorm) is a widely adopted technique that enables faster and more stable training of deep neural networks (DNNs). Despite its pervasiveness, the exact reasons for BatchNorm’s effectiveness are still poorly understood. The popular belief is that this effectiveness stems from controlling the change of the layers’ input distributions during training to reduce the so-called “internal covariate shift”. In this work, we demonstrate that such distributional stability of layer inputs has little to do with the success of BatchNorm. Instead, we uncover a more fundamental impact of BatchNorm on the training process: it makes the optimization landscape significantly smoother. This smoothness induces a more predictive and stable behavior of the gradients, allowing for faster training. These findings bring us closer to a true understanding of our DNN training toolkit.  
**Link**: arxiv.org/abs/1805.11604  


Shar1 [22 hours ago]  
**Title** : DropBlock: A regularization method for convolutional networks  
**Abstract** : Deep neural networks often work well when they are over-parameterized and trained with a massive amount of noise and regularization, such as weight decay and dropout. Although dropout is widely used as a regularization technique for fully connected layers, it is often less effective for convolutional layers. This lack of success of dropout for convolutional layers is perhaps due to the fact that activation units in convolutional layers are spatially correlated so information can still flow through convolutional networks despite dropout. Thus a structured form of dropout is needed to regularize convolutional networks. In this paper, we introduce DropBlock, a form of structured dropout, where units in a contiguous region of a feature map are dropped together. We found that applying DropbBlock in skip connections in addition to the convolution layers increases the accuracy. Also, gradually increasing number of dropped units during training leads to better accuracy and more robust to hyperparameter choices. Extensive experiments show that DropBlock works better than dropout in regularizing convolutional networks. On ImageNet classification, ResNet-50 architecture with DropBlock achieves 78.13% accuracy, which is more than 1.6% improvement on the baseline. On COCO detection, DropBlock improves Average Precision of RetinaNet from 36.8% to 38.4%  
**Link** : https://arxiv.org/abs/1810.12890  

I think this is gonna be quite effective during training.  


Shar1 [22 hours ago]  
**Title**: Rethinking ImageNet Pre-training  
**Abstract** : We report competitive results on object detection and instance segmentation on the COCO dataset using standard models trained from random initialization. The results are no worse than their ImageNet pre-training counterparts even when using the hyper-parameters of the baseline system (Mask R-CNN) that were optimized for fine-tuning pre-trained models, with the sole exception of increasing the number of training iterations so the randomly initialized models may converge. Training from random initialization is surprisingly robust; our results hold even when: (i) using only 10% of the training data, (ii) for deeper and wider models, and (iii) for multiple tasks and metrics. Experiments show that ImageNet pre-training speeds up convergence early in training, but does not necessarily provide regularization or improve final target task accuracy. To push the envelope we demonstrate 50.9 AP on COCO object detection without using any external data---a result on par with the top COCO 2017 competition results that used ImageNet pre-training. These observations challenge the conventional wisdom of ImageNet pre-training for dependent tasks and we expect these discoveries will encourage people to rethink the current de facto paradigm of 'pre-training and fine-tuning' in computer vision.
**Link** : https://arxiv.org/abs/1811.08883  

ajit [18 hours ago]  
**Title**: 
Classification with Costly Features using Deep Reinforcement Learning  
**Abstract**:  
We study a classification problem where each feature can be
acquired for a cost and the goal is to optimize a trade-off be-
tween the expected classification error and the feature cost. We
revisit a former approach that has framed the problem as a se-
quential decision-making problem and solved it by Q-learning
with a linear approximation, where individual actions are ei-
ther requests for feature values or terminate the episode by
providing a classification decision. On a set of eight problems,
we demonstrate that by replacing the linear approximation
with neural networks the approach becomes comparable to the
state-of-the-art algorithms developed specifically for this prob-
lem. The approach is flexible, as it can be improved with any
new reinforcement learning enhancement, it allows inclusion
of pre-trained high-performance classifier, and unlike prior art,
its performance is robust across all evaluated datasets  
**Link**:    
https://arxiv.org/pdf/1711.07364v2.pdf (edited)  


ajit [16 hours ago]  
**Title**:  
Strike (with) a Pose: Neural Networks Are Easily Fooled
by Strange Poses of Familiar Objects  
**Abstract**:    
Despite  excellent  performance  on  stationary  test  sets,
deep  neural  networks  (DNNs)  can  fail  to  generalize  to
out-of-distribution  (OoD)  inputs,  including  natural,  non-
adversarial ones, which are common in real-world settings.
In this paper, we present a framework for discovering DNN
failures that harnesses 3D renderers and 3D models.  That
is, we estimate the parameters of a 3D renderer that cause
a target DNN to misbehave in response to the rendered im-
age.  Using our framework and a self-assembled dataset of
3D objects, we investigate the vulnerability of DNNs to OoD
poses of well-known objects in ImageNet.  For objects that
are readily recognized by DNNs in their canonical poses,
DNNs incorrectly classify 97% of their pose space. In addi-
tion, DNNs are highly sensitive to slight pose perturbations.
Importantly, adversarial poses transfer across models and
datasets.  We find that 99.9% and 99.4% of the poses mis-
classified by Inception-v3 also transfer to the AlexNet and
ResNet-50 image classifiers trained on the same ImageNet
dataset, respectively, and 75.5% transfer to the YOLOv3 ob-
ject detector trained on MS COCO.  
**Link**:  
https://arxiv.org/pdf/1811.11553.pdf  
