# Notes for Teaching Day

Quinn -- Opinizers
    Opinizers reduese the weighnt of the links between the the neurons layers
    if greadiont is going up then redust the weight

Ted -- Freezing
    Freeing the weight on the Old CNN Traing files layers 
    Freezing is hiperpriaminder

Porter -- Image Data pipeline
    Augmentation: Rotataing the image becase the image could be taken at different againlses
        Rotation is random from 90 - 15 degress each time ** Do more recherch
    Normalization: The model see numbers not an image, So there is a rand of mean and std for each layers so the layer doent crash the model
    Train vs. calidation transforms: Traing transform is random, Validation is deterministic
        Different jobs
    ImageFolder class indices: Classes are classafide athabeticaly A,B,C,... 
    
    ColorJitter so the colors dont get washed out


----------------------------------------------------------

# Note for Sharing Implementation_Guides

Quinn
    Design Decision: Save the results in both a Json and the files 
    Question: I Dont have any Question
    Takaway: I need to be more presise in my destions, like deciding the range for tunine

    Notes: Broke done the frezzing into 3 different stages start with everthing frozzen then slowing unfrezzing. 


Ted
    Design Decision: Resizing and croping for transfomation
    Question: I Dont have any questions
    Takaway: Metrixs are vary up in the air alowing for good tuning

    Notes: Checked for different modes effesiontyes, Backbone will all be frozen then sloly unfreeze. Using Adam W for opinizer. Vary opean for metrixs. Learnung rate


Porter
    Design Decision: implmented Learnig rates for both phase one and two
    Question: Why would you want to unfrezzs all the layers? Why do you plain to start with only on classifers first? 
    Takaway: Clear Guide with headers, decisiotons are well orginzed

    Note: Two pahases feature extraction and fine-tuning. For every Decision the stakeholder impact was though of which was vary smart desition. Freezing will have two pahased only classifiers unfrozon and all NOT frozzen. Guide is vary well orginzed and easy to read. Did a lot of thinking though the prosses and guide for the project


Over all Takaway
    Over all I think we all did a good Job, we all had simulary desinges, with all wanting to playaround with frezzing. Presionaly I need to add all the little detatils, so the metrics for each desidions. Like each bach would be 32 and why I would i want this. 


