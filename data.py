import pandas as pd

parliament_results = {
    1973: 5.0,
    1977: 1.9,
    1981: 4.5,
    1985: 3.7,
    1989: 13.0,
    1993: 6.3,
    1997: 15.3,
    2001: 14.6,
    2005: 22.1,
    2009: 22.9,
    2013: 16.3,
    2017: 15.3,
    2021: 11.7,
    2025: 23.8,
}

local_results = {
    1975: {"municipal": 0.8, "county": 1.4},
    1979: {"municipal": 1.9, "county": 2.5},
    1983: {"municipal": 5.3, "county": 6.3},
    1987: {"municipal": 10.4, "county": 12.3},
    1991: {"municipal": 6.5, "county": 7.0},
    1995: {"municipal": 10.5, "county": 12.0},
    1999: {"municipal": 12.1, "county": 13.4},
    2003: {"municipal": 16.4, "county": 17.9},
    2007: {"municipal": 17.5, "county": 18.5},
    2011: {"municipal": 11.4, "county": 11.8},
    2015: {"municipal": 9.5, "county": 10.2},
    2019: {"municipal": 8.2, "county": 8.7},
    2023: {"municipal": 11.4, "county": 12.5},
}

poll_data = [
    {"date": "Jan 10", "FrP": 23.4},
    {"date": "Feb 10", "FrP": 22.7},
    {"date": "Mar 10", "FrP": 23.7},
    {"date": "Apr 10", "FrP": 23.9},
    {"date": "May 10", "FrP": 22.6},
    {"date": "Jun 10", "FrP": 23.5},
    {"date": "Jul 10", "FrP": 22.3},
    {"date": "Aug 10", "FrP": 22.8},
    {"date": "Sep 10", "FrP": 22.9},
    {"date": "Jan 11", "FrP": 23.8},
    {"date": "Feb 11", "FrP": 23.2},
    {"date": "Mar 11", "FrP": 22.2},
    {"date": "Apr 11", "FrP": 18.9},
    {"date": "May 11", "FrP": 19.2},
    {"date": "Jun 11", "FrP": 19.1},
    {"date": "Jul 11", "FrP": 19.7},
    {"date": "Aug 11", "FrP": 17.2},
    {"date": "Sep 11", "FrP": 16.5},
    {"date": "Jan 12", "FrP": 13.7},
    {"date": "Feb 12", "FrP": 14.4},
    {"date": "Mar 12", "FrP": 16.3},
    {"date": "Apr 12", "FrP": 17.3},
    {"date": "May 12", "FrP": 16.0},
    {"date": "Jun 12", "FrP": 16.2},
    {"date": "Jul 12", "FrP": 18.4},
    {"date": "Aug 12", "FrP": 17.3},
    {"date": "Sep 12", "FrP": 16.6},
    {"date": "Jan 13", "FrP": 15.9},
    {"date": "Feb 13", "FrP": 15.6},
    {"date": "Mar 13", "FrP": 16.8},
    {"date": "Apr 13", "FrP": 16.5},
    {"date": "May 13", "FrP": 16.0},
    {"date": "Jun 13", "FrP": 16.4},
    {"date": "Jul 13", "FrP": 16.2},
    {"date": "Aug 13", "FrP": 15.2},
    {"date": "Sep 13", "FrP": 15.9},
    {"date": "Jan 14", "FrP": 13.6},
    {"date": "Feb 14", "FrP": 13.3},
    {"date": "Mar 14", "FrP": 14.3},
    {"date": "Apr 14", "FrP": 14.0},
    {"date": "May 14", "FrP": 13.7},
    {"date": "Jun 14", "FrP": 13.8},
    {"date": "Jul 14", "FrP": 12.3},
    {"date": "Aug 14", "FrP": 13.9},
    {"date": "Sep 14", "FrP": 14.1},
    {"date": "Jan 15", "FrP": 11.3},
    {"date": "Feb 15", "FrP": 11.2},
    {"date": "Mar 15", "FrP": 10.4},
    {"date": "Apr 15", "FrP": 11.0},
    {"date": "May 15", "FrP": 12.0},
    {"date": "Jun 15", "FrP": 12.3},
    {"date": "Jul 15", "FrP": 12.8},
    {"date": "Aug 15", "FrP": 12.7},
    {"date": "Sep 15", "FrP": 12.2},
    {"date": "Jan 16", "FrP": 16.4},
    {"date": "Feb 16", "FrP": 16.7},
    {"date": "Mar 16", "FrP": 16.7},
    {"date": "Apr 16", "FrP": 16.4},
    {"date": "May 16", "FrP": 15.5},
    {"date": "Jun 16", "FrP": 15.9},
    {"date": "Jul 16", "FrP": 15.0},
    {"date": "Aug 16", "FrP": 14.4},
    {"date": "Sep 16", "FrP": 14.5},
    {"date": "Jan 17", "FrP": 13.9},
    {"date": "Feb 17", "FrP": 14.0},
    {"date": "Mar 17", "FrP": 12.6},
    {"date": "Apr 17", "FrP": 12.8},
    {"date": "May 17", "FrP": 13.0},
    {"date": "Jun 17", "FrP": 13.4},
    {"date": "Jul 17", "FrP": 13.3},
    {"date": "Aug 17", "FrP": 14.5},
    {"date": "Sep 17", "FrP": 14.6},
    {"date": "Jan 18", "FrP": 14.2},
    {"date": "Feb 18", "FrP": 13.7},
    {"date": "Mar 18", "FrP": 15.2},
    {"date": "Apr 18", "FrP": 15.8},
    {"date": "May 18", "FrP": 14.9},
    {"date": "Jun 18", "FrP": 13.4},
    {"date": "Jul 18", "FrP": 14.8},
    {"date": "Aug 18", "FrP": 12.8},
    {"date": "Sep 18", "FrP": 13.9},
    {"date": "Jan 19", "FrP": 12.3},
    {"date": "Feb 19", "FrP": 11.3},
    {"date": "Mar 19", "FrP": 11.3},
    {"date": "Apr 19", "FrP": 10.6},
    {"date": "May 19", "FrP": 12.2},
    {"date": "Jun 19", "FrP": 10.4},
    {"date": "Jul 19", "FrP": 8.2},
    {"date": "Aug 19", "FrP": 10.2},
    {"date": "Sep 19", "FrP": 11.5},
    {"date": "Jan 20", "FrP": 12.8},
    {"date": "Feb 20", "FrP": 15.0},
    {"date": "Mar 20", "FrP": 12.9},
    {"date": "Apr 20", "FrP": 11.5},
    {"date": "May 20", "FrP": 9.9},
    {"date": "Jun 20", "FrP": 11.6},
    {"date": "Jul 20", "FrP": 10.6},
    {"date": "Aug 20", "FrP": 11.0},
    {"date": "Sep 20", "FrP": 12.6},
    {"date": "Jan 21", "FrP": 9.5},
    {"date": "Feb 21", "FrP": 8.5},
    {"date": "Mar 21", "FrP": 9.7},
    {"date": "Apr 21", "FrP": 9.9},
    {"date": "May 21", "FrP": 10.3},
    {"date": "Jun 21", "FrP": 10.2},
    {"date": "Jul 21", "FrP": 9.9},
    {"date": "Aug 21", "FrP": 10.3},
    {"date": "Sep 21", "FrP": 11.3},
    {"date": "Jan 22", "FrP": 11.9},
    {"date": "Feb 22", "FrP": 11.5},
    {"date": "Mar 22", "FrP": 11.5},
    {"date": "Apr 22", "FrP": 12.5},
    {"date": "May 22", "FrP": 12.9},
    {"date": "Jun 22", "FrP": 14.3},
    {"date": "Jul 22", "FrP": 13.6},
    {"date": "Aug 22", "FrP": 13.8},
    {"date": "Sep 22", "FrP": 14.0},
    {"date": "Jan 23", "FrP": 12.9},
    {"date": "Feb 23", "FrP": 12.7},
    {"date": "Mar 23", "FrP": 12.1},
    {"date": "Apr 23", "FrP": 12.2},
    {"date": "May 23", "FrP": 12.1},
    {"date": "Jun 23", "FrP": 12.4},
    {"date": "Jul 23", "FrP": 14.2},
    {"date": "Aug 23", "FrP": 12.7},
    {"date": "Sep 23", "FrP": 12.9},
    {"date": "Jan 24", "FrP": 13.1},
    {"date": "Feb 24", "FrP": 13.8},
    {"date": "Mar 24", "FrP": 14.8},
    {"date": "Apr 24", "FrP": 15.2},
    {"date": "May 24", "FrP": 17.1},
    {"date": "Jun 24", "FrP": 17.2},
    {"date": "Jul 24", "FrP": 16.5},
    {"date": "Aug 24", "FrP": 16.4},
    {"date": "Sep 24", "FrP": 18.9},
    {"date": "Jan 25", "FrP": 24.4},
    {"date": "Feb 25", "FrP": 24.6},
    {"date": "Mar 25", "FrP": 22.7},
    {"date": "Apr 25", "FrP": 20.0},
    {"date": "May 25", "FrP": 20.6},
    {"date": "Jun 25", "FrP": 21.0},
    {"date": "Jul 25", "FrP": 21.5},
    {"date": "Aug 25", "FrP": 21.2},
    {"date": "Sep 25", "FrP": 21.0},
]

# Create a DataFrame
# df = pd.DataFrame(poll_data)


# # Function to parse date strings
# def custom_date_parser(date_str):
#     parts = date_str.split(" ")
#     month_str = parts[0]
#     year_str = parts[1]

#     # Prepend '20' to make a four-digit year
#     year_full = "20" + year_str

#     return f"{month_str} 1 {year_full}"


# df["parsed_date"] = df["date"].apply(custom_date_parser)
# df["date"] = pd.to_datetime(df["parsed_date"], format="%b %d %Y")
# df.drop(columns=["parsed_date"], inplace=True)

# # Set the date as the index for time series analysis
# df.set_index("date", inplace=True)

# # Save the preprocessed DataFrame to a CSV file
# df.to_csv("frp_poll_data_processed.csv")

# # Print the head of the processed DataFrame to show the new format
# print("First 5 rows of the processed DataFrame:")
# print(df.head())
# print("\nDataFrame info to show the date format:")
# df.info()
