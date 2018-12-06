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

Rahul [15 hours ago]  
**Title**: Image-to-Image Translation with Conditional Adversarial Networks  

**Abstract**: We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.  
**Link**: https://arxiv.org/abs/1611.07004  


z0k [6 hours ago]  
**Title**: Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control  
**Abstract**: Deep reinforcement learning (RL) algorithms can learn complex robotic skills from raw sensory inputs, but have yet to achieve the kind of broad generalization and applicability demonstrated by deep learning methods in supervised domains. We present a deep RL method that is practical for real-world robotics tasks, such as robotic manipulation, and generalizes effectively to never-before-seen tasks and objects. In these settings, ground truth reward signals are typically unavailable, and we therefore propose a self-supervised model-based approach, where a predictive model learns to directly predict the future from raw sensory readings, such as camera images. At test time, we explore three distinct goal specification methods: designated pixels, where a user specifies desired object manipulation tasks by selecting particular pixels in an image and corresponding goal positions, goal images, where the desired goal state is specified with an image, and image classifiers, which define spaces of goal states. Our deep predictive models are trained using data collected autonomously and continuously by a robot interacting with hundreds of objects, without human supervision. We demonstrate that visual MPC can generalize to never-before-seen objects---both rigid and deformable---and solve a range of user-defined object manipulation tasks using the same model.  
**Arxiv link**: https://arxiv.org/abs/1812.00568  
