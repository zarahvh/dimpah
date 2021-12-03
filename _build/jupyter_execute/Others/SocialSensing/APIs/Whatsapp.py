# Whatsapp

There has been an interesting report recently on how WhatsApp data can be used in the study of radical networks: http://www.huffingtonpost.com/entry/german-study-finds-radicalized-muslims-have-little-actual-knowledge-of-islam_us_5967f362e4b03389bb163c58 

A group of German scholars at the Universities of Bielefeld and Osnabrück analyzed 5,757 WhatsApp messages found on a phone seized by police following a terrorist attack in the spring of 2016. The messages were exchanged among 12 young men involved in the attack. The attack itself was not identified in the report. 

This tutorial is motivated by this study and shows another type of interaction with social media, where we can only download the data and not directly access it through an API.

In this tutorial, we construct the graph of distribution of messages over a given time period using your own Whatsapp data. Note that Whatsapp does not have an API (http://mashable.com/2015/03/25/whatsapp-developers-api/). But you can download your messages (or anybody else's) and analyse the download. This is an alternative method in order to work with data that you have no other access to. 


We will use prepared sample data but you can also take a look at how to download your Whatsapp messages: https://www.whatsapp.com/faq/en/general/23753886

import matplotlib.pyplot as plt
import pandas as pd

whatsapp = pd.read_csv("whatsapp-output.csv")
whatsapp.head()

whatsapp.loc[whatsapp['Speaker'] == 'MESSAGE','sender'] = 'Her' 
whatsapp.loc[whatsapp['Speaker'] == 'GRT','sender'] = 'Me' 
whatsapp.head()

whatsapp = whatsapp.drop(columns=['Speaker', 'SentenceNo', 'SequenceNo'])
whatsapp.head()

Now let's plot two overlapping histograms to compare 'Her' and 'Me' sending messages.

whatsapp['datetime'] = pd.to_datetime(whatsapp['Date'].astype(str) + ' ' +whatsapp['Time'].astype(str))
whatsapp['datetime'] = whatsapp['datetime'].dt.floor('Min')

her = whatsapp.loc[whatsapp['sender'] == 'Her']
me = whatsapp.loc[whatsapp['sender'] == 'Me']

herplot = her.plot(kind='bar', x='datetime', y='message_length', color='blue', position=1, label='Her')
me.plot(kind='bar', x='datetime', y='message_length', color='red', ax=herplot, position=0, label='Me')

plt.show()

That's it. Blue is 'Her' sending whatsapp messages, red is 'Me'. It seems that 'Me' is chatting much more ...