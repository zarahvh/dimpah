# Predicting Modelling

Today, we will learn that machine learning is much less scary than science fiction will want to us to believe. This is not because we have benevolent machines, which only want our best, but simply because these machines are quite far away from living their own life without our input, as Skynet manages in 'Terminator' or the Machine in 'Person of Interest'. For the time being, machines still learn best when provided with human input. Furthermore, machines learn in most applications not because they want to start to understand the meaning of life and find out that humans are obstacles to true life, but because they learn to complete a particular task. Machines learn to be part of the workbenches of digital productions.

It is maybe less a link to artificial intelligence in science fiction than the fact that machines learn from our examples and need to be fed with large amounts of data to learn that makes machine learning an ethically difficult endeavour. Machine learning demands ever more data. Most aspects of our lives are recorded in vast data stores that are easily accessible to machines. Governments, businesses and individuals are recording and reporting all manners of information from the monumental to the mundane. As long as the activities can be transformed into digital formats, you can be certain that somebody will record it. 

In such a world, machines learn by consuming data and humans continuously add new digital methods of machine learning that can exploit this data. These can be some of the statistical methods we have already met or more advanced ones, we will meet today. The digital methods we learn about today have in common that they aim to predict new observations from old observations. They are all empirical and predictive using models.

Machine learning algorithms are all around you. They have tried to predict the outcomes of elections and referenda, can identify spam messages, predict crime and natural disasters, target donors and voters as well as finally have learned how to drive cars. Recently, they got it wrong quite often: http://www.kdnuggets.com/2016/11/trump-shows-limits-prediction.html

Many stories are told about the uses and abuses of machine learning. You can find some in the readings. Given how much machine learning is now part of our everyday life, it is maybe surprising that there are not even more stories. 

We also still lack an ethics of machine learning, which is developing so fast that it is difficult for laws and norms to stay up to date. There is, for instance, an on-going debate how biased machine learning algorithms are with regard to race and gender. Machine learning also has made it possible to identify people based on the region they live, the products they buy, etc. 

As a machine learning practitioner, you are often required to exclude revealing data that is ethically problematic, but this is not an easy task, as sometimes the connections are not obvious and might only be revealed after you have trained the machine to learn. 

## Background 1: The Data Science Process

### Social and cultural analytics and its data

Just like humans, machines use data to generalize. They abstract the data and develop its underlying principles, because humans tell them how. In the words of machine learning, machines form a model, which assigns meaning and represents knowledge. The model summarizes the data and makes explicit patterns among data. 

There are many different types of models. We have already seen some and others you will know from school. Models can be (statistical) equations, figures like graphs or trees, rules or clusters. Machines don't choose the type of models, we choose them for them when analysing the task at hand and the available data. 

The computer learns to fit the model to the data in a process called training. However, computational modelling does not end here. We also need to test the model in a separate testing process. The model thus does not include anything else but what can be found in the data already. It can nevertheless be interesting, as the model might surface connections that we did not recognize before. Newton discovered gravity this way by fitting a series of equations (a model) to observations of falling apples – if the myth is to be believed. Gravity was always there but it was observed for the first time in a model. 

Modelling is far from perfect. It generally involves some kind of bias or systematic error. Newton's laws of gravity are not as universal as he thought they would be. 

Errors like this do not have to be a bad thing, because they can lead the computer to be able to learn a better model, correcting previous mistakes. But generally, bias is to be avoided. Your reading includes the example where a machine learning algorithms learned to discriminate wolves and huskies from a series of online pictures. It achieved excellent performance until somebody found out that the decision was often based on whether snow can be found in the pictures’ background.

All learning has weaknesses and is biased in a particular way. Researchers are still looking for the universal model that is better than the rest of them but will probably never find it. Therefore, it is really important to understand how a model can overcome bias. This is the purpose of testing it on new data.

Unfortunately, especially in our domain of social and cultural analytics, models often fall short of desirable performance. Humans are difficult for computers and the data they produce and can be judged by is very noisy. This means that social and cultural data includes many errors because observations have not been measured correctly or maybe they are simply impossible to measure. How do you quantify, for instance, love? It seems impossible, but online match making agencies still make a business out of predicting love. 

Humans are also inconsistent and report data wrongly. Finally, especially in history we simply do not have data for all time periods or if we have data, it will include many missing values or will be badly captured according to diverse and sometimes contradictory standards. Often, the records have simply been lost. 

A final complication with data in social and cultural analytics that has only recently emerged is the limited access we have to the data. Because it is so valuable, it is kept behind the walls of company servers and is not shared.

So, machine learning is not artificial intelligence yet but a laborious collaboration between humans and machines that involves trying models and fighting with (bad) data. Otherwise, machine learning is a process that consists of a series of repeatable steps, which we will learn about today. In today's reading Schutt and O'Neil (2013), have given us an excellent overview of the art of data science.



import cv2
  
# Save image in set directory
# Read RGB image
img = cv2.imread('C:\\process.png') 
  
# Output img with window name as 'image'
cv2.imshow('result', img) 



