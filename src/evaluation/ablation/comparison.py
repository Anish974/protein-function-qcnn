import pandas as pd


class AblationComparison:
    """
    Creates publication-ready comparison tables.
    """

    @staticmethod
    def summarize(results):

        dataframe = pd.DataFrame(results)

        dataframe = dataframe.sort_values(

            by="Accuracy",

            ascending=False,

        )

        return dataframe