---
title: "Chess Games Analysis"
date: 2020-02-15
classes: wide
excerpt: "This paper is about analyzing a dataset of chess games"
---

### Problem Statement
I chose a data set of 20k chess games collected from the website lichess.org. I found this particularly interesting as I used to be a professional chess player, about 15 years ago. I still play as a hobby. This data set has 16 different columns including the Time controls used, White and Black players and their ratings, Opening used in the game, winner and the winning method, and also the entire moves set.  

I am hoping to do a basic analysis of this data set and make some predictions about winning chances based on the color, user’s rating level, time controls used, opening used etc. I would also try to correlate some of these fields and to make predictions such as, if certain openings work better in certain time controls and if some openings are more suited for experienced players than for beginners etc.

### Conclusion
I used logistic regression and trained a model with a moderate level of accuracy. You can find the full report below.

### Important Files
Full report - [Chess Games Analysis](https://github.com/dasun27/DSC/blob/master/files/Chess%20Games%20Analysis.docx)  
Dataset - [Lichess.org](https://www.opendatanetwork.com/dataset/opendata.ramseycounty.us/baxh-fa9y)
