# Fluent: An AI Augmented Writing Tool for People who Stutter

## Overview

Stuttering is a speech disorder which impacts the personal and professional lives of millions of people worldwide. To save themselves from the stigma and embarrassment, people who stutter (PWS) may adopt different strategies to conceal their stuttering. One of the common strategies is word substitution where an individual avoids saying a word they might stutter on and use an alternative instead. Research has shown that this process itself can cause stress and add more burden. In this work, we present Fluent, an AI augmented writing tool which assists a PWS in writing scripts which they speak fluently. Fluent embodies a novel Active learning based method of identifying words an individual might struggle pronouncing. Such words are highlighted in the interface. On hovering over any such word, Fluent presents a set of alternative words which have similar meaning but are easier to speak. The user is free to accept or ignore these suggestions. Based on such user actions (feedback), Fluent continuously evolves its classifier to better suit the personalized needs of each user. Using a simulation study, we demonstrate the effectiveness of our tool to learn user's unique needs. Such a tool can be beneficial for certain important life situations like giving a talk, presentation, etc.

## Installation Instructions

- Clone this repo

- Install Dependencies using: pip install -r req.txt

- python -m spacy download en_core_web_sm

- Run python app.py

- Browse localhost:3999
