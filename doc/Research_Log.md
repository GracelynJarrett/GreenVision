# Transfer Learning Mechanics

## What layers of efficienctnet learned during imagenet training
    - Early layers learn low-level features such as Edges, colores and simple textures. These are generic acorss almost all images
    - Middle layers learn mid-level features like patternes, repeating textures and simple objct parts
    - Deep layers learn high-level class-specific features such as objects shapes (dog face, car parts) and Semantic meaning

## Fine-tuning vs Feature Extraction vs Training from Scratch
    - Feature Extraction: the most common starting point. Use EfficentNest as-is, only train final classifer layer. Its sues fore samll dataset and Similar domain
    - Fine-tuning: Unfeexe some deeper layers and train them. Use them in Medium size dataset, why data is similar byt not identical to ImageNet, and when you want better accuracy
    - Training from Scratch: Randomly initialize all weights, bessed to use with Massive datasets (100k+ images) anc completely different data

## What "Freezing" means + backpropagation behavior
    Freexing prevents layers weights from updateing Preventing destroying useful pre-trained knoledge and reduces overfitting. This makes training faster and more stable
    - backpropagation behavior
        Normal: Forward pass (compute output) -> Compute loss -> Backprop: Compute gradients and Update wights
        
        Frozen layer: Fowardpass -> backprop: No gradient computed and wights NOT updated
    
## Why pre-trained layers need lower learning rates
    Pre-trained layers already contain valuabel knowledge, you want to adjust slightly not overwrite
    - Problem with high learing rates : Large gradient updates and pre-trained knowledge get destroyed quickly [This is called Catstrophic forgetting]
    - Why new Higher Learing rates: they start randomly initalized and need big updates to learn quickly

----------------------------------------------------------------------------------


# CNN Architecture for image classification

## How Convolutional layers extract spatial features
    A conolution layer applies small filters called Kernels that slides across the image. Each Kernels learns to detect speific visual patterns like edge detectores, color transitions and texture patterns. Each layer builds upon the orevious layer. Unlike regula neural networks CNNS presever pisal position relationships, meaning pixels close to =gether stay meaningful.

## What pooling does and Why its used
    Pooling reduces the spatial size of the feature map, reduceing computation. Pooling keeps imprtant features but ignores small variarions, also helps prevent overfitting.
        - Most common Pooling is max or Avg

        Input (4×4) →  MaxPool(2×2)
        [ 1 3 2 1 ]    →  [ 3 4 ]
        [ 5 6 1 2 ]       [ 6 8 ]
        [ 0 2 3 4 ]
        [ 1 7 2 8 ]

## Receptive fiels -what it means
    - Definition: How much of the original image a neuron can "see"
   
    Each layer "see more" the deeper you go relaying on the prvase layer. Small receptive fiels fines more details whyle Large receptive fiedls sees more of the global structure

    Layer 1: looks at small patch
    Layer 2: combines multiple patches
    Layer 3: combines larger regions...

## How to calculate output dimenstions
    - Covolution output formula for one dimenstions

                (N - F + 2P)             Where: N = input size  F = filter size
                ____________  + 1               P = padding     S = stride
                      S

EX: Input: 224×224  Filter: 3×3  Stride: 1  Padding: 1      [(224−3+2×1)/1]+1=224


    - Pooling output formula

                (N - F)
               _________ +1
                   S

EX: Inout 224   Pool: 2   Stride: 2         [(224-2)/2] + 1 = 112


Full Pipline EX:
Input: 224×224

Conv (3x3, stride=1, pad=1) → 224×224
MaxPool (2x2)              → 112×112
Conv (3x3)                 → 112×112
MaxPool                    → 56×56


----------------------------------------------------------------------------------


# Image Date Pipelines

## Data augmentation Strategies
    - Label-preserving transformations: This does not change hte meaning of the image so the label stays correct
            Feometic Transforms: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
    - Color/lighting transfoms: helps modes handale different lighting conditions and camera differences
            ColorJitter (brightness, contrast, saturation), TandomGrayscale, Slight noise
    - Cropping /Scaling
            RandomResixedCrop, Resize + CenterCrop

    DANGEROUS: these can change or hide things making labels incorrect
        Examples for this project:
            ❌ Large random erasing (may remove disease spots)
            ❌ Extreme cropping (removes infected region)
            ❌ Heavy blur (destroys texture patterns)
            ❌ Extreme color shifts (turns green → purple)
            ❌ Random mixup (blending two different diseases—advanced technique only)
    Rull of thumn Ask "Would a human still correctly label this image?" if NO -> don't use it


## Why normalization matters
    Normalization Scales pixel vaues, makeing stable training which keeps values in a consistent range and prevents exploding/vanshing gradients. When using EffcientNet, it expects speciffic normalization vaues, matching pre-trained modesl

    Wrong Normalization couse Model confuseion where the ubout districution doesnt't  match traing data, and features become meaningless. Could case Traing instablity lossing manu Not decrease and ump wildly. Also poor accuracy may accoure even if the traiing "works".

    Formula:
                X - mean
        Xnorm = _________
                   std

## Train vs Validation transfoms
    - training transforms purpose is to imorove generalization and include random augmentation 
    - Validation transforms have no randomness and only resizing + normalization. This insuers consistenty and fair evaluation

    Mix them up
        - Validation uses agumentation will lean to results changeing every run and accuracy becomes noisy
        - training has no augmentation the preforms will be poorly on new images and the model will become overfited quickly

## How ImageFolder assigns Class indices

    ImgeFolders should be sorted alphabetically, this determines labels, not folder order on the disk

        - Why this matters 
            During training model learns oupt neuron 0 ="healthy", Output neuron 1 = "powdery_mildew
        If mapping is wrong -> predictions are wrong

        SAVE MAPPING

    Folder structure Example for project:
    data/
        train/
            healthy/
            rust/
            powdery_mildew/


----------------------------------------------------------------------------------

# Deep learning training practices

## How to detect overfitting in a CNN
    - What is overfitting: your model memorized training data but doesn't generalize to new images

    -How to detect it;
        compare traing accuracy/loss and Vaalidation accuracy/loss

        If Both are improcing togethere there is no overfitting
        IF Train accuracu increassing but validation accuracy stall or decreasses your overfitting


## Dropout and weight decay
    Main defenses against overfitting are Dropout and Weight Decay
        - Drop out randomly "Turns off" neurons during traing, preventing neurons from relying on specific others, forceing the model to learn robust patterns
        - Weight Decay Penalizes larege weights, Meaning large weights -> complex and overfit models wile small weights -> smoother and simpler modles
            Formula Loss+λ∑w2


## Batch size tradeoffs
    Batch size is the numer of images processed at once
    
    -Small bactchsize
        Pros: Better generalization, less memory usege and more "noisy" gradients -> helps escape oveerfitting
        Cons: Slower traing and less stable updates

    -Large batch size
        Pros: Faster traing and more stable gradients
        Cons: Requires more GPU memory and can generalize worse

## When to stop training (early stopping)
    After some point traing improves and validation gets wores. Solution stop traing when validation stops improving. To do this moniter the train and Val loss if Val loss incresed stop, Wait a few epochs before stopping, when the model stops it means the model has leaned s much as it can 

----------------------------------------------------------------------------------


# Experiment tracking and model serving

## Logging CNN training runs with MLflow
    Use MLflow and Pytorch together.
    Log hyperparamets, metrics per epoch and the trained model

    Useing both Mlfow and pytorch together will compare experiments easily, showing tracked improvements scienfiically and helps debugs

## Saving and loading Pytorch codels correctly
    There are two main approaches 
        -1 Save state_dict
            This aproach is lightweight, flexible and works across environments
        -2 Save entire models 



## Image preprocessing at inference
    Core rule: inference preprocessing MUST match validation preprocessing EXACTLY
         modesl learn on Resize → CenterCrop → ToTensor → Normalize. If this is change input distrucution changes and the model predictions become unreliable

        What breaks things:
        Using different normalization values, Not resizing correctly, Skipping cropping, Using different image size

## FastAPI image upload handling
    -Opion 1 Multipart upload: client sends dile directly ; Pros: efficient Handles lare giles well, standard approach
    -Opion 2 Base64 encoding: image is converted to string and sent to Json ; Cons: ~33% larger payload, slower, more processing overhead




# What I could reacher more and teach the class

Receptive fields
    Definition: a receptive field that is part of the input image that influences a single neuron's output 

    Receptive fields is the size of the image that a noded sees at once. Receptive fields are always compared to the original image size. Each layer will increass nodes view by 2. 

    If we have pooling it will doubal the view increass by dubble

    J is normaly 1 but pooling will dubbal that to 2. J will then stay 2 untill it reaches a nother pooling wich will jump it up to 4

    Formula
    RF_new = RF_old + (kernel_size - 1) * J_old
    J_new  = J_old * stride