# Build an ML Pipeline for Short-Term Rental Prices in NYC
You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

This repository is available at
https://github.com/TomasCajan/build-ml-pipeline-for-short-term-rental-prices

- This is my solution of the second MLOps nanodegree capstone project.
- It contains the whole pipeline, currently released in version 1.0.1
- Please find related wandb workspace at https://wandb.ai/tomas-cajan23/nyc_airbnb?nw=nwusertomascajan23
- I made two commits, one original and one after fixing the cleaning step accoring to project rubric
- I have changed the MLproject files syntaxes to cope with WIN CMD, since that is what I am used to work with at my work
- In the last trial I made, the pipeline passed without problem with the sample2 file
- To run it, use following command
  
mlflow run https://github.com/TomasCajan/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"

Commentary

My job normaly includes data science and training smaller scale of different machine learning models, but up to this point my work did not focus too much on productionizing the code. After doing all it took to complete this project, I am quite confident in taking my job responsibilities to a next level. It was quite intense experince putting the pipeline together - to cut through all the version missmatches, spotting false tips in the course and solving conflicts between linux and windows commands, all eventuelly leading to an elegant functional and repeatable pipeline. I spent quite a few hours on it, more that I would like to admit, but the sense of accomplishment is strong here, no regrets. It was significantly more difficult than the first capstone project, so I am curious about how will the next one look like, the bar is quite high at this moment. While I am not going to continue tuning up this project, I am totally going to implement the principles I have learned here, in my own model training scripts, to make them more reproductioble and professional overal.
