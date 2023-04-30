from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] > 4 else 0

class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data), index_col=0)
        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        # _df = pd.DataFrame({'review': reviews, 'label': labels})
        one_hot_cols = ['division_name', 'department_name']
        self.outcome_col = 'label'
        # one hot encode the categorical columns and drop missing values
        df_new = pd.get_dummies(
            self.df.dropna(subset=one_hot_cols + [self.outcome_col])[one_hot_cols],
            columns=one_hot_cols, drop_first=True
        )
        # select features to be used in the model
        features_cols = ['age']
        for oc in one_hot_cols:
            features_cols.extend([col for col in df_new.columns if col.startswith(oc)])
        self.features_cols = features_cols
        _df = self.df.dropna(subset=one_hot_cols + [self.outcome_col]).join(df_new, how='left')
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')
        self.next(self.baseline)

    @step
    def baseline(self):
        """Compute the baseline"""
        from sklearn.metrics import roc_auc_score, accuracy_score
        from sklearn.linear_model import LogisticRegression
        self.model = (
            LogisticRegression(class_weight='balanced')
            .fit(
                X=self.traindf[self.features_cols],
                y=self.traindf[self.outcome_col]
            )
        )
        # TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        self.base_acc = accuracy_score(
            y_true=self.valdf[self.outcome_col],
            y_pred=self.model.predict(self.valdf[self.features_cols])
        )
        self.base_rocauc = roc_auc_score(
            y_true=self.valdf[self.outcome_col],
            y_score=self.model.predict_proba(self.valdf[self.features_cols])[:,1]
        )
        self.next(self.end)
        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0. 
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        df_false_positives = self.valdf[
            (self.valdf[self.outcome_col] == 0) & (self.model.predict(self.valdf[self.features_cols]) == 1)
        ]
        current.card.append(Table.from_dataframe(df_false_positives))
        
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1. 
        # TODO: display the false_negatives dataframe using metaflow.cards
        df_false_negatives = self.valdf[
            (self.valdf[self.outcome_col] == 1) & (self.model.predict(self.valdf[self.features_cols]) == 0)
        ]
        current.card.append(Table.from_dataframe(df_false_negatives))

if __name__ == '__main__':
    BaselineNLPFlow()
