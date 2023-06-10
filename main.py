# This is a sample Python script.
# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    import pandas as pd
    import requests
    import matplotlib

    matplotlib.use('TkAgg')
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Fetch data from API
    url = "https://disease.sh/v3/covid-19/countries"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.head()

    # Remove irrelevant columns
    columns_to_keep = ["country", "cases", "todayCases", "deaths", "todayDeaths", "recovered", "active"]
    df = df[columns_to_keep]

    # Remove null values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.head()

    # Histogram of cases
    fig, ax = plt.subplots()
    sns.histplot(df["cases"], bins=20, ax=ax)
    st.pyplot(fig)
    st.text("Cases")
    st.text("Frequency")
    st.title("Distribution of Cases")

    # Scatter plot of cases vs. deaths
    fig1, ax = plt.subplots()
    sns.scatterplot(x=df["cases"], y=df["deaths"], ax=ax)
    st.pyplot(fig1)
    st.text("Cases")
    st.text("Deaths")
    st.title("Cases vs. Deaths")

    # Drop the 'country' column
    df.drop("country", axis=1, inplace=True)
    df.drop("todayCases", axis=1, inplace=True)

    # Select numerical columns for scaling
    numerical_cols = ["cases", "deaths", "recovered", "active"]
    # Split the data into train and test sets
    X = df.drop("cases", axis=1)
    y = df["cases"]

    # Scale the numerical columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Scale the numerical columns

    X_train, X_test, y_train, y_test = train_test_split(df[numerical_cols], y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    # Define the prediction function

    def predict_cases(cases_today, deaths_today, recovered, active):
        # Scale the input values
        input_data = scaler.transform([[cases_today, deaths_today, recovered, active]])

        # Make the prediction
        predicted_cases = model.predict(input_data)

        return predicted_cases[0]


    # Create a user-friendly interface with Streamlit
    st.title("COVID-19 Case Predictor")

    # Add input fields for the user
    cases_today = st.number_input("Cases today (0 to infinity)", min_value=0)
    deaths_today = st.number_input("Deaths today (0 to infinity)", min_value=0)
    recovered = st.number_input("Recovered (0 to infinity)", min_value=0)
    active = st.number_input("Active (0 to infinity)", min_value=0)

    # Make a prediction when the button is clicked
    if st.button("Predict"):
        prediction = predict_cases(cases_today, deaths_today, recovered, active)
        st.success("Predicted Cases: {}".format(prediction))
        # Calculate and display the mean squared error
        mse = mean_squared_error(y_test, y_pred)
        st.info("Mean Squared Error: {}".format(mse))
    # Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
