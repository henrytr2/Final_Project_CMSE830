import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

### setting up 
st.set_option('deprecation.showPyplotGlobalUse', False)

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'times', sans-serif;
			}
			</style>
			"""

st.markdown(streamlit_style, unsafe_allow_html=True)

st.image('hi.jpg')
cols=st.columns(3,gap='large')
with cols[0].expander("CNN:"):
    st.video("https://www.youtube.com/watch?v=wO-DAdZX59w",start_time=0)
with cols[1].expander("Consumer Reports:"):
    st.video("https://www.youtube.com/watch?v=DBTmNm8D-84",start_time=0)
with cols[2].expander("WSJ:"):
    st.video("https://www.youtube.com/watch?v=t7RiQbjlOfQ",start_time=0)
tab1, tab2 , tab3 , tab4, tab5, tab6, tab7= st.tabs(['Insurance Dataset', 'Data Distributions', "Relationship to Charges","Feature Analysis", "Linear and Lasso","Prediction","Conclusions"])
### Tab 1, introduction
with tab1:
    st.title("Introduction")
    st.write("In a recent survey conducted by Gallup and West Health, an estimated 112 million Americans (44%) struggle to pay for [healthcare](https://www.westhealth.org/press-release/112-million-americans-struggle-to-afford-healthcare/#:~:text=WASHINGTON%2C%20D.C.%20%E2%80%94%20Mar.,is%20not%20worth%20the%20cost)."
            " National health spending is over 4 trillion in this country, and current projections indicate it will continue to grow at an annual rate of 5.4%, topping $6.2 trillion by 2028. "
            "Furthermore, [The Wall Street Journal](https://www.wsj.com/health/healthcare/health-insurance-cost-increase-5b35ead7) has recently reported that insurance rates are set to increase more than the projected rate, at 6.5%. This would mark on of the biggest price increase in years. "
            )
    st.write("This application is designed for indivuduals who wish to understand factors that lead to insurance charges. The application investigates possible trends for insurance charges. Towards this goal, several models were trained. ")
    data = pd.read_csv("insurance.csv")

    col3, col1,col2=st.columns(3,gap='small')
    button1 = col1.checkbox("Show Statistics")
    if button1 ==True:
        st.write(data.describe())
        st.caption("**Table 1:** Summary Statistics of the Insurance Dataset")
    button2 = col2.checkbox('Show Insurance Dataset')
    if button2==True:
        st.table(data.head())
        st.caption("**Table 2:** The First Five Rows of the Insurance Dataset")
    button3 = col3.checkbox('Column Information')
    if button3 == True:
        column_info = {
        'age': 'Age of the insured person',
        'sex': 'Gender of the insured person (male or female)',
        'BMI': 'Body Mass Index of the insured person',
        'children': 'Number of children or dependents covered by the insurance',
        'smoker': 'Whether the insured person is a smoker (Yes or No)',
        'region': 'Region in the United States of the insured person'
        }
        for col, desc in column_info.items():
            st.write(f"- **{col}**: {desc}")


bio= st.sidebar.checkbox("**Author Bio**")
if bio:
    st.sidebar.image('TH.jpg')
    st.sidebar.write("Trent Henry is a Ph.D student in the Communicative Sciences and Disrders Department at Michigan State University. He recieved his B.A. Psychology from MSU in 2022. His work focuses on Voice Disorders and how Machine Learning can lead to their more accurate diagnosis")
    st.sidebar.write(" In his free time Trent likes to read comic books, watch movies and listen to the Beatles.")


###### End Tab 1 

###### Tab 2 Visulization 
with tab2:
    st.title("Distriutions of Variables")
    st.write("First, We investigate the disributions for serveral variables in the dataset. In this tab, we have included the distributions for age, whether or not someone is a smoker, and BMI."
            )
    def age_group(X):
        if X in range(18,20):
            return '18-20'
        if X in range(20,30):
            return '20-30'
        if X in range(30,40):
            return '30-40'
        if X in range(40,50):
            return '40-50'
        elif X in range(50,60):
            return '50-60'
        else:
            return '60+'
    data['age_group']=data['age'].apply(age_group)


    fig1 = px.histogram(data, x="age_group", title='Age Distribution of Health Insurers',
                    labels={'age_group': 'Age Group', 'count': 'Count'},color='age_group')
    fig1.update_layout(xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False,title='Frequency'))
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={"age_group": ['18-20', '20-30', '30-40', '40-50', '50-60', '60+']}
    st.plotly_chart(fig1)
    st.caption("**Figure 1:** Distribution of ages. Ages were grouped based on ages 18-20, 20-30, 30-40, 40-50, 50-60, and 60 or above respectively")
    # fig2
    fig2 = px.histogram(data, x="smoker", title='Distribution of Smokers in the Dataset', 
                    labels={'smoker': 'Smoker Status', 'count': 'Count'},color='smoker')
    fig2.update_layout(xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False,title='Frequency'))
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={"smoker": ['Yes', 'No']}
    st.plotly_chart(fig2)
    st.caption("**Figure 2:** Distribution of smokers and non-smokers in the dataset")

    # fig3
    fig3 = px.histogram(data, x='bmi', marginal='box', nbins=64, title='Distribution of BMI')
    fig3.update_layout(xaxis=dict(showgrid=False,title='BMI'),
            yaxis=dict(showgrid=False,title='Frequency'))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**Figure 3:** Distribution of BMI among the dataset")

   

 
with tab3:
    st.title('Relationship to Insurance Charges')
    data = pd.read_csv("insurance.csv")
    st.write("Next, we have provided interactive visulizations to provide the user with insights as to features that impact the rate of insurance charges.")
    # color options
    x_option = st.selectbox('Select X Option:', ['bmi', 'age'])
    color_option = st.selectbox('Select Color Option:', data.columns)

# scatter plot of the data and colors 
    scatter_fig = px.scatter(data, x=x_option, y="charges", color=color_option,
                            hover_data=['bmi', 'smoker', 'children'])

    # bmi = BMI 
    scatter_fig.update_layout(
        xaxis=dict(title='BMI' if x_option == 'bmi' else x_option.capitalize()),
        yaxis=dict(title='Insurance Charges'),
    )

        # bmi = BMI 

    scatter_fig.update_layout(
        title_text=f'Relationship between {"BMI" if x_option == "bmi" else x_option.capitalize()} and Insurance Charges',
        title_font_size=20
    )

    # Show the scatter plot
    st.plotly_chart(scatter_fig)

    #   bmi = BMI 

    st.caption(f"**Figure 4:** Scatter plot showing the relationship between {'BMI' if x_option == 'bmi' else x_option.capitalize()} and charges with color indicating {'BMI' if color_option == 'bmi' else color_option.capitalize()}.")

    # Create grouped features
with tab4:
    st.title("Feature Importantance Analysis")
    st.write("When trying to predict insurance costs, there are several features that need to be considered. We have used two models, a random forest regressor model and a gradient boosting model. This is done to identify if there are certain features that impact insurance charges more than others. ")

# Load data
    data = pd.read_csv("insurance.csv")

    # Changing features by grpouping ages bmi etc
    bin_edges_bmi = [15, 30, 45, 60]
    bin_labels_bmi = ['Small', 'Medium ', 'Large']

    bin_edges_child = [-1, 2, 7]
    bin_labels_child = ['0-2', '+2']

    bin_edges_age = [17, 32, 48, 70]
    bin_labels_age = ['Yound', 'Middle Aged', 'Elder']

    data['grouped_bmi'] = pd.cut(data['bmi'], bins=bin_edges_bmi, labels=bin_labels_bmi)
    data['grouped_child'] = pd.cut(data['children'], bins=bin_edges_child, labels=bin_labels_child)
    data['grouped_age'] = pd.cut(data['age'], bins=bin_edges_age, labels=bin_labels_age)
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region', 'grouped_bmi', 'grouped_child', 'grouped_age'], drop_first=True)

    
    selected_model = st.selectbox('Select Model:', ['Random Forest Regressor', 'Gradient Boosting Regressor'], key='model_selectbox')

    

    # Train-test split
    X = data.drop(['charges', 'bmi', 'children', 'age'], axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # selection of models
    if selected_model == 'Random Forest Regressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif selected_model == 'Gradient Boosting Regressor':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)



    # model fit 
    model.fit(X_train, y_train)

    # feature importance
    feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    if feature_importances is not None:
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        
        fig = px.bar(importance_df, x='Feature', y='Importance', title=f'Feature Importance for {selected_model}', 
                    labels={'Importance': 'Feature Importance'}, color='Importance', color_continuous_scale='turbo')

        
        st.plotly_chart(fig)
        st.caption(f"**Figure 5:** Bar chart  showing the feature important for the {' Forest Regressor' if selected_model== 'Random Forest Regressor' else 'Gradient Boosting Regressor'}")

        

        # Print feature importance
        st.write("Feature Importance:")
        st.write(importance_df)
        st.caption(f"**Table 3:** Ranking of the feature important for {selected_model} ")

    # Make predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    st.write(f'R-squared (Accuracy): {r2:.2f}')



with tab5:
    st.title("Linear Regression and Lasso Analysis")
    st.write("After determing the features that contrbute the most to the insurance charges, perform linear regression in an effort to predict insurance charges for an indivual. After selection of the model, r-squared values are shown. ")
    data = pd.read_csv("insurance.csv")
    # Feature engineering
    bin_edges_bmi = [15, 30, 45, 60]
    bin_labels_bmi = ['Small', 'Medium', 'Large']

    bin_edges_child = [-1, 2, 7]
    bin_labels_child = ['0-2', '+2']

    bin_edges_age = [17, 32, 48, 70]
    bin_labels_age = ['Yound', 'Middle Age', 'Elder']

    data['grouped_bmi'] = pd.cut(data['bmi'], bins=bin_edges_bmi, labels=bin_labels_bmi)
    data['grouped_child'] = pd.cut(data['children'], bins=bin_edges_child, labels=bin_labels_child)
    data['grouped_age'] = pd.cut(data['age'], bins=bin_edges_age, labels=bin_labels_age)
    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region', 'grouped_bmi', 'grouped_child', 'grouped_age'], drop_first=True)

    # User input for model selection
    selected_model = st.selectbox('Select Model:', [ 'Linear Regression', 'Lasso'], key='model_selectbox2')

    # Train-test split
    X = data.drop(['charges', 'bmi', 'children', 'age'], axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_model == 'Linear Regression':
        model = LinearRegression()
    elif selected_model == 'Lasso':
        model = Lasso(alpha=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics

    r2 = r2_score(y_test, y_pred)

    # Display metrics
    st.write(f'R-squared (Accuracy): {r2:.2f}')


    scatter_fig = px.scatter(data, x=y_test, y=y_pred, trendline='ols', labels={'x': 'Actual Charges', 'y': 'Predicted Charges'})
    scatter_fig.update_traces(marker=dict(size=8, opacity=0.6), selector=dict(mode='markers'))
    st.plotly_chart(scatter_fig)
    st.caption('**Figure 6:** Linear Regression Plot')
with tab6:
    insurance = pd.read_csv("insurance.csv")
    insurance = pd.get_dummies(insurance, columns=['smoker', 'region','sex'], drop_first=True)




# Split the dataset into features
# not valid names?
    X = insurance[['age', 'bmi', 'children', 'smoker_yes', 'sex_male', 'region_northwest', 'region_southeast', 'region_southwest']]
    y = insurance['charges']

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # User inputs for prediction
    st.title("Predict Insurance Charges")
    st.write("This tab allows the user to input data relating to themsleves for prediction of insurance chages. The prediction is based on our previously created linear regression model.")
    user_age = st.slider('Age:', min_value=18, max_value=100, value=30)
    user_bmi = st.slider('BMI:', min_value=15, max_value=50, value=25)
    user_children = st.slider('Number of Children:', min_value=0, max_value=10, value=0)
    user_smoker = st.radio('Smoker:', ['Yes', 'No'])
    user_sex = st.radio('Gender:', ['Male', 'Female'])

    # Encode 'smoker' and 'sex' based on user input
    user_smoker_encoded = 1 if user_smoker == 'Yes' else 0
    user_sex_encoded = 1 if user_sex == 'Male' else 0

    # Predict insurance charges
    user_data = [user_age, user_bmi, user_children, user_smoker_encoded, user_sex_encoded, 0, 0, 0]  # For 'None' hue
    predicted_charges = model.predict([user_data])
    st.write(f"# ***Your Predicted Insurance Charges: ${predicted_charges[0]:.2f}***")
with tab7:
    st.title("Conclusions")
    st.write("From the analysis performed, we can see several interesting points.")
    st.write("When we first looked at the EDA and data distributions, we saw that perhaps whether or not someone was a smoker played a significant role in the insurance charges. To validate this claim, we performed feature analysis using  Grandient Regressor (.87 r-squared) and Random Forest (.84 r-squared) models. In both cases the model indicated that smoking does play a significant role in insurance charges.")
    st.write("After feature analysis, we then created a linear model with .77 r-squared that user may implement to understand factors for their insurance charges.")

    st.title("Further Work")
    st.write("Although this datset provides information regarding insurance charges, several key factors are left out. These include things such as race, SES, or martial status. Future work could be performed with these variables included in the dataset. This would provide a more robust model. ")


