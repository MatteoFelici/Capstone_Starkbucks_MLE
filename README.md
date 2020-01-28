# Capstone Project for Udacity's Machine Learning Engineer Nanodegree
# Starbucks App data

This is my work for the final work of the Nanodegree. The goal of the project is to analyze
historical data about Starbucks' app usage in order to develop an algorithm
that finds the most suiting offer type for each customer.

We have 3 datasets:
- *Portfolio:* it contains the list of all available offers to propose to the customer.
Each offer can be a *discount*, a *BOGO (Buy One Get One)* or *Informational* (no real offer)
- *Profile:* this is the list of all customers that interacted with the app.
- *Transcript:* this dataset contains the list of all actions on the app relative to special offers,
plus all the customers’ transactions.

I divided the process into 3 phases, each one developed in a specific Notebook:
- **Data preparation:** first look at the data, then join all the datasets to recreate the customer's journey
- **Feature Engineering:** enrichment of the dataset, creating new features from available data
- **Modeling:** development of 2 distinct algorithms (for each type of offer), then choosing the best one
in terms of performance

You can find the description of the entire process in detail in the **Capstone_project.pdf** report file.
