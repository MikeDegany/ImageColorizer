**TOPIC :-** _Image colorization and resolution improvement using Generative Adversarial Networks_

**TEAM :-**

_Rana Ghoneim (_[_RanaGhoneim@my.unt.edu_](mailto:RanaGhoneim@my.unt.edu)_)_

_Saja Al Kawari (_[_SajaAlkarawi@my.unt.edu_](mailto:SajaAlkarawi@my.unt.edu)_)_

_Mike Degany (_[_Mike.Degany@gmail.com_](mailto:Mike.Degany@gmail.com)_)_

_Shridhar Kshirsagar (_[_ShridharKshirsagar@my.unt.edu_](mailto:ShridharKshirsagar@my.unt.edu)_)_

**GITHUB LINK :-** [_ **https://github.com/Dehghanni/ImageColorizer** _](https://github.com/Dehghanni/ImageColorizer)

**VIDEO LINK :-**

[_ **https://drive.google.com/file/d/1gOcKDdFrtbxQMgvAjloGHm6YL6v7EFpa/view?usp=sharing** _](https://drive.google.com/file/d/1gOcKDdFrtbxQMgvAjloGHm6YL6v7EFpa/view?usp=sharing)

**ABSTRACT**

The tradition of preserving and transferring knowledge from one generation to the next generation exists even today. Most of the time this knowledge happens to be in the form of visuals, images of certain old sculptures, scriptures etc. Quality of these images is directly influenced by the technology available during these generations. To overcome this problem and to minimize the loss of this vital visual information, we propose an Artificially Intelligent software able to convert the existing low quality and low resolution images into high quality and high resolution ones. Many technologies to transform low quality images into high quality ones currently exist. We limit the scope of our project to work only with black and white images, increase the resolution and colorize them. Our objective is to extend the currently existing state of art to obtain more accurate results.

**INTRODUCTION**

Color perception is an important aspect of human perception that aids in decision-making. We can use it to distinguish items and elicit feelings. It&#39;s a powerful tool that&#39;s been utilized in the creative world for a long time. Nonetheless, comprehension is essential. It takes time and experience to master the use of color. It necessitates theoretical understanding. And comprehension of the world on a semantic level, because the choice of Colors has the ability to drastically affect the mood of a situation. The retrieval of lost data in the colorization of grayscale photographs is a difficult challenge. Multiple results will be judged correct, although not all will reflect the image&#39;s principal meaning. We explore the issues of image colorization in this project and offer a deep learning model for colorizing (c) photography. Our technique is trained on a dataset developed as part of the project and is based on generative adversarial network architecture various images of houses, trees, streets, glaciers mountains, and other objects are captured in our dataset. It is a dynamic image collection. Keras and Tensorow were used to implement the system. Finally, the model was validated and evaluated using industry-standard criteria.

**PROBLEM SPECIFICATION**

  1. **Dataset:**

Colorizing black and white photos is one of the most interesting applications of deep learning. Several years ago, this task required a lot of human input and hardcoding, but now, thanks to AI and deep learning, the entire process can be completed end-to-end. We use the COCO dataset for our project. Our dataset contains images of houses, trees, streets, glaciers, mountains, and other objects in two folders.

The application of Artificial Intelligence that we plan to implement for this project is Generative Adversarial Networks. Generative Adversarial Networks are based on the adversarial concept of two AI agents or algorithms working together, with one trying to better the results produced by other one. We intend to use the transfer learning concept of Machine Learning, i.e., loading a pre-trained GAN network and re-training it again with our input and selectively freezing some layers in the network.

![](RackMultipart20220511-1-dc2fhq_html_b0353816e011a9ae.png)

GAN Architecture

  1. **Problem Analysis:**

The application of Artificial Intelligence that we plan to implement for this project is Generative Adversarial Networks. Generative Adversarial Networks are based on the adversarial concept of two AI agents or algorithms working together, with one trying to better the results produced by the other one. We intend to use the transfer learning concept of Machine Learning, loading a pre-trained GAN network and re-training it again with our input and selectively freezing some layers in the network.

Most of the recent developments deal either with the colorization of an image or increasing the resolution of images. We propose to integrate these parts of improving the image resolution and colorizing it later

**DESIGN AND MILESTONES**

1. **Proposed Method :-** Based upon the aforementioned design, this project will be using conditional General Adversarial Networks in order to generate colored images out of low quality black and white pictures. therefore. this project uses conditional adversarial networks, in which two losses are used: L1 loss, which makes it a regression task, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner (by assigning the outputs a number indicating how &quot;real&quot; they look!).

Generative adversarial networks (GANs) provide a way to learn deep representations without extensively annotated training data. They achieve this through deriving backpropagation signals through a competitive process involving a pair of networks. [1]

In other word, GAN is a framework for estimating generative models via an adversarial process, in which two models should be simultaneously trained: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. [2]

In contrary to many learning methods that need huge amount of data or long training times to train the model from scratch for such problems, this project uses a GAN model with specific architectures, loss functions, training strategies with an efficient strategy to train such a model, using the latest advances in deep learning, on a rather small dataset and with really short training times.

Generative Adversarial Networks (GANs) are an emerging technique for both semi-supervised and unsupervised learning. They achieve this through implicitly modelling high-dimensional distributions of data. they can be characterized by training a pair of networks in competition with each other. A common analogy, apt for visual data, is to think of one network as an art forger, and the other as an art expert. The forger, known in the GAN literature as the generator, G, creates forgeries, with the aim of making realistic images.
 The expert, known as the discriminator, D, receives both forgeries and real (authentic) images, and aims to tell them apart Both are trained simultaneously, and in competition with each other.[1]

The diagram below shows the general structure of GAN that has been used in this project.

![](RackMultipart20220511-1-dc2fhq_html_3fef0eaf91ae17ff.png)

As GAN&#39;s can be used in a variety of applications, the input, synthetic and real data can be anything. In the following we will discuss about the input, real and synthetic data in this project.

In this case, the generator model takes a grayscale image (1-channel image) and produces a 2-channel image, a channel for \*a and another for \*b. The discriminator takes these two produced channels and concatenates them with the input grayscale image and decides whether this new 3-channel image is fake or real. Of course, the discriminator also needs to see some real images (3-channel images again in Lab color space) that are not produced by the generator and should learn that they are real[3]. The L\*a\*b color space will be discussed later.

1. **Data Processing :-** The dataset used for this project is the famous COCO dataset. It is a large image dataset with images belonging to 80 different classes. COCO dataset is mainly used for supervised Machine Learning Tasks like object detection, image classification, image captioning, etc. For this project, it has been used as a source of various different images for the GAN to learn from.

Below are some of the images from the dataset.

![](RackMultipart20220511-1-dc2fhq_html_68c9281b1591ab21.png)

The classes of the dataset have not been used. Steps involved in Data Processing are as follows:-

- We choose 10,000 random images from the dataset. This is done because the dataset is too huge to train the GAN with all the images and with limited hardware resources it would take quite a while to train. Further, we split the images into training and validation datasets with the ratio 80:20.
- Usually the image data is represented in color space in RGB format. There are three pixels which represent the amount of red, green and blue in each pixel. As we look at this problem as a regression problem, the RGB channel has been converted into LAB color space. LAB color space is again a tuple representing three values. L represents the lightness of the image and when we visualize this channel we get a black and white image. A and B channels contain information about how much red-green and green-blue each pixel holds. The problem of colorization has been solved by using the L channel values as input (Grayscale) and trying to predict the A and B channel values and later combine all three to get a colored image, which makes this problem a regression problem as &#39;L&#39;, &#39;A&#39; and &#39;B&#39; values are continuous in nature. But if you use RGB, you have to first convert your image to grayscale, feed the grayscale image to the model and hope it will predict 3 numbers for you which is a way more difficult and unstable task due to the many more possible combinations of 3 numbers compared to two numbers. If we assume we have 256 choices (in a 8-bit unsigned integer image this is the real number of choices) for each number, predicting the three numbers for each of the pixels is choosing between 256³ combinations which is more than 16 million choices, but when predicting two numbers we have about 65000 choices.

![](RackMultipart20220511-1-dc2fhq_html_f2b92183e961281b.jpg)

Fig. RGB color channels

![](RackMultipart20220511-1-dc2fhq_html_f7da347e0248a08e.jpg)

Fig. LAB color channels

- As the image size varies through the dataset, the images have been resized to a fixed size of 256x256. Additionally, we flip some random images horizontally to bring about a little augmentation in the training data.
- Finally we create an object data loader, which loads a batch of 16 images at a time to pass into the Generator.

1. **Experimental Settings :-**

- **Generator -** As a Generator of the Generative Adversarial Network, a Convolutional Neural Network called UNet has been used. UNet is a typical pix to pix image converter network whose input is a three channeled image and for this project, two values (A and B color channels) have been predicted. UNet comprises a total of 20 convolution layers, wherein each layer stacks up a feature map and in the middle of the UNet we get a 1x1 image, which is later sampled up and outputs two feature map which contain the A and B channel values. The output can be considered as an image which contains only A and B channel values.

![](RackMultipart20220511-1-dc2fhq_html_3b34a0d8ed6a975e.png)

Fig. UNet Architecture

- **Discriminator -** Discriminator&#39;s purpose is to produce confidence on how real the image looks. A Patch discriminator has been implemented in the project which outputs confidence in the form of a numerical value for every patch of 70x70 pixels.

1. **Validation Methods :-** As previously established, the problem has been approached as a regression problem, traditional regression problem validation methods have been used.

- **GAN Loss :-** A GAN specific loss function called as a vanilla model has been implemented.A vanilla model with a prediction for real image as 1.0 and fake image as 0.0 has been used as the overall GAN loss in the architecture.
- **L1 Loss :-** An extra loss function called L1 loss has been used in order to further help the model in producing more real looking images.

Below are some of the results produced after 20 epochs of training.

![](RackMultipart20220511-1-dc2fhq_html_d2299b9fc8549594.png)

Fig. Results after 20 epochs

![](RackMultipart20220511-1-dc2fhq_html_120f8148e26cfbfc.png)

Fig. Loss Curve for first 7 epochs

From the loss curve it can be observed that the loss for the Generator and Discriminator model goes on decreasing with every epoch.

**LIMITATIONS**

- Generator model takes a long time to train as the model is trained from scratch without any previous knowledge of the images. This can be overcome by using a pretrained model like a Resnet or an Inception V3.
- Due to the limited hardware resources the dataset needed to be cut down to 10000 data instances. This can be overcome by more efficient and faster hardware resources.

**PROJECT MANAGEMENT:**

The project tasks were evenly distributed throughout the group members, with a focus on working on topics that matched their strengths.

1. **Implementation status report :-**

The Generator network is trained enough to give decent results. It tries to colorize the black and white images. With availability of better and faster hardware, the training time can be reduced considerably.

1. **Work completed :-**

The two main objectives of the project were colorizing the image and preserving its resolution while doing so. The objectives were met successfully.

1. **Description :-**

We used Generative Adversarial Networks (GAN) as an AI application. We choose 10,000 images randomly from our dataset and we implemented Machine Learning&#39;s transfer learning principle, importing a pre-trained GAN network, and retraining it with our input while freezing specific layers selectively. We run all the codes, implement diagrams, made the presentation and the video demo, and finally wrote the report.

1. **Responsibility :-**

| **Group Member** | **Task** |
| --- | --- |
| Shridhar Kshirsagar | Project report, Code and Video Demo |
| Mohammad Dehghanitezerjani | Project report, Diagrams, and Presentation |
| Saja Al Karawi | Project Report and Diagrams |
| Rana Ghoneim | Project Report and Presentation |

1. **Contributions :-**

Shridhar Kshirsagar was in charge of writing the codes and making the video demo for the project and also reviewing the project report and determining which diagrams had chosen to be used in the report. In addition, Mohammad Dehghanitezerjani was in charge of implementing the diagrams, writing parts of the presentation and parts of the project report. Also, Saja Al Karawi was in charge of the implementing of other diagrams and writing other parts of the project. Furthermore, Rana Ghoneim was in charge of writing parts of the report as well and writing other parts of the presentation too.

1. **Issues/Concerns :-**

1. All images from the dataset could not be used for training due to memory and hardware limitations.

2. Model took a long time to give some satisfactory results.

3. As pytorch was used, model history was not saved hence, had to plot the loss curve separately for Generator and Discriminator.

**FUTURE WORK**

- In the next phase the lower resolution images can be refined and converted into high resolution images and then cascaded to the image colorization GAN&#39;s generator model.
- Hence, our input image will be a lower resolution grayscale image which will first be upsampled to a higher resolution image and then colorized.

**References:**

[1] Creswell, A., White, T., Dumoulin, V., Arulkumaran, K., Sengupta, B. and Bharath, A.A., 2018. Generative adversarial networks: An overview. _IEEE Signal Processing Magazine_, _35_(1), pp.53-65.

[2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Generative adversarial nets. _Advances in neural information processing systems_, _27_.

[3] Moein Shariatnia, Image Colorization with U-Net and GAN Tutorial. shorturl.at/lrNY4

[https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image](https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image)
