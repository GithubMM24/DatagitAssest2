import pandas as pd
import re

class DataAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.context = {}

        # Normalized column lookup
        self.columns = {col.lower(): col for col in df.columns}

        # Numeric columns
        self.numeric_columns = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]


    # Find column from query
    def find_column(self, query):
        for key, col in self.columns.items():
            if key in query:
                return col
        return None


    # Main processor
    def process_query(self, query: str):
        query = query.lower()
        df = self.df.copy()

        # value filtering (categorical)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                for val in df[col].dropna().unique():
                    if str(val).lower() in query:
                        df = df[df[col] == val]
                        self.context[col] = val

        # range filter
        between = re.search(
            r'between\s+(\d+\.?\d*)\s+(?:to|and)\s+(\d+\.?\d*)',
            query
        )
        if between:
            low, high = map(float, between.groups())
            col = self.find_column(query)
            if col in self.numeric_columns:
                df = df[(df[col] >= low) & (df[col] <= high)]


        # greater / less than filter FEATURE
        gt = re.search(r'(greater than|above)\s+(\d+\.?\d*)', query)
        lt = re.search(r'(less than|below|under)\s+(\d+\.?\d*)', query)

        col = self.find_column(query)
        if col in self.numeric_columns:
            if gt:
                df = df[df[col] > float(gt.group(2))]
            if lt:
                df = df[df[col] < float(lt.group(2))]

        # sorting logic 
        sort_order = None
        if "ascending" in query or "asc" in query:
            sort_order = True
        if "descending" in query or "desc" in query:
            sort_order = False

        if sort_order is not None:
            # Priority 1: "based on / by <column>"
            based_on = re.search(r'(based on|by)\s+([\w\s]+)', query)
            if based_on:
                sort_col = self.find_column(based_on.group(2))
                if sort_col in self.numeric_columns:
                    df = df.sort_values(by=sort_col, ascending=sort_order)
            else:
                # Priority 2: any numeric column mentioned
                col = self.find_column(query)
                if col in self.numeric_columns:
                    df = df.sort_values(by=col, ascending=sort_order)
                # Priority 3: fallback → first numeric column
                elif self.numeric_columns:
                    df = df.sort_values(
                        by=self.numeric_columns[0],
                        ascending=sort_order
                    )

        # top-N / bottom-N
        top_n = re.search(r'top\s+(\d+)', query)
        if top_n:
            n = int(top_n.group(1))
            col = self.find_column(query)
            if col in self.numeric_columns:
                df = df.sort_values(
                    by=col,
                    ascending="lowest" in query or "min" in query
                ).head(n)

        # max / min
        if "highest" in query or "maximum" in query:
            col = self.find_column(query)
            if col in self.numeric_columns:
                return {
                    "intent": "MAX",
                    "result": df[col].max()
                }

        if "lowest" in query or "minimum" in query:
            col = self.find_column(query)
            if col in self.numeric_columns:
                return {
                    "intent": "MIN",
                    "result": df[col].min()
                }


        # average
        if "average" in query or "mean" in query:
            col = self.find_column(query)
            if col in self.numeric_columns:
                return {
                    "intent": "AVG",
                    "result": round(df[col].mean(), 2)
                }


        # count
        if "count" in query:
            return {
                "intent": "COUNT",
                "result": len(df)
            }


        # column select
        selected_columns = []
        for key, col in self.columns.items():
            if key in query:
                selected_columns.append(col)

        selected_columns = list(dict.fromkeys(selected_columns))

        
        if "summary" in query or "what's going on" in query:
            return {
                "intent": "DATASET_SUMMARY",
                "result": {
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "numeric_columns": self.numeric_columns,
                    "column_names": df.columns.tolist()
                }
            }

        if "sample" in query:
            return {
                "intent": "SAMPLE_DATA",
                "result": df.head(5)
            }

        # ----------------------------------
        # ----------------------------------
        if selected_columns:
            return {
                "intent": "FILTERED_DATA",
                "result": df[selected_columns]
            }

        return {
            "intent": "FILTERED_DATA",
            "result": df
        }
