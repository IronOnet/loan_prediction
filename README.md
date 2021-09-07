# Loan Prediction With Machine Learning

Using machine learning to predict whether bank loans will be repaid or not.


The values you are prompted for are:

| Value                   | Description |
| :---                    | --- |
| project_name            | Loan prediction | 
| project_description     | Predicting the probability of loan repaiement with machine learning | 
| repo_name               | https://github.com/IronOnet/loan_prediction |
| conda_name              | env |
| package_name            | Scikitlearn, Pytorch | 
| author                  | IronOnet | 
| open_source_license     | M.I.T | 
| devops_organisation     | An Azure DevOps organisation. Leave blank if you aren't using Azure DevOps | 



You are now ready to get started, however you should first create a new github repository for your new project and add your 
project using the following commands (substitute myproject with the name of your project and REMOTE-REPOSITORY-URL 
with the remote repository url).

    cd myproject
    git init
    git add .
    git commit -m "Initial commit"
    git remote add origin REMOTE-REPOSITORY-URL
    git remote -v
    git push origin master



## Contributing to This Project
Contributions to this Project are greatly appreciated and encouraged.

To contribute an update simply:
* Submit an issue describing your proposed change to the repo in question.
* The repo owner will respond to your issue promptly.
* Fork the desired repo, develop and test your code changes.
* Check that your code follows the PEP8 guidelines (line lengths up to 120 are ok) and other general conventions within this document.
* Ensure that your code adheres to the existing style. Refer to the
   [Google Cloud Platform Samples Style Guide](
   https://github.com/GoogleCloudPlatform/Template/wiki/style.html) for the
   recommended coding standards for this organization.
* Ensure that as far as possible there are unit tests covering the functionality of any new code.
* Check that all existing unit tests still pass.
* Edit this document and the template README.md if needed to describe new files or other important information.
* Submit a pull request.


### Project development environment
To develop this template further you might want to setup a virtual environment

#### Setup using
```

python -m venv venv
```

#### Activate environment
Max / Linux
```
source env/bin/activate
```

Windows
```
env\Scripts\activate
```

#### Install Dependencies
```
pip install -r requirements.txt
```