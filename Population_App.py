# Імпортуємо всі необхідні бібліотеки
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objects as go
from numerize import numerize
import warnings

warnings.filterwarnings("ignore")

# Налаштування сторінки Streamlit
st.set_page_config(layout="wide")

# Завантаження даних
countries_pop = pd.read_csv(r'datasets/Countries_Population_final.csv')
countries_name = pd.read_csv(r'datasets/Countries_names.csv')

# Заголовок панелі
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    pass
with col2:
    st.info('# :blue[POPULATION PREDICTION SYSTEM]')
with col3:
    pass

# Вибір країни та року
col1, col2 = st.columns(2)
with col1:
    option = st.selectbox('PLEASE SELECT ANY COUNTRY', sorted(countries_name['Country_Name']))
    year = st.text_input('PLEASE ENTER YEAR', '2030')

    if year.isnumeric():
        target_year = int(year)

        # Вибір незалежних і залежних змінних
        X = countries_pop['Year']
        y = countries_pop[option]

        # Розбиття на тренувальний і тестовий набори
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        # Функція для поліноміальної регресії
        def create_polynomial_regression_model(degree):
            poly_features = PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train)
            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, Y_train)
            r2_test = r2_score(Y_test, poly_model.predict(poly_features.fit_transform(X_test)))

            years_range = np.arange(X.min(), target_year + 1).reshape(-1, 1)
            predictions_poly = poly_model.predict(poly_features.fit_transform(years_range))
            return years_range, predictions_poly, r2_test

        # Функція для лінійної регресії
        def create_linear_regression_model():
            lin_model = LinearRegression()
            lin_model.fit(X_train, Y_train)
            r2_test = r2_score(Y_test, lin_model.predict(X_test))

            years_range = np.arange(X.min(), target_year + 1).reshape(-1, 1)
            predictions_lin = lin_model.predict(years_range)
            return years_range, predictions_lin, r2_test

        # Генерація прогнозів
        years_poly, pred_poly, accuracy_poly = create_polynomial_regression_model(2)
        years_lin, pred_lin, accuracy_lin = create_linear_regression_model()

        # Відображення результатів
        pred_pop_poly = numerize.numerize(pred_poly[-1])
        pred_pop_lin = numerize.numerize(pred_lin[-1])

        st.write("#### :green[POLYNOMIAL REGRESSION ACCURACY: ] " + str(int(accuracy_poly * 100)) + '%')
        st.write("#### :green[LINEAR REGRESSION ACCURACY: ] " + str(int(accuracy_lin * 100)) + '%')
        st.write("#### :green[COUNTRY:  ] " + option.upper())
        st.write("#### :green[YEAR:  ] " + year)
        st.write("#### :green[PREDICTED POPULATION (POLYNOMIAL):  ] " + pred_pop_poly)
        st.write("#### :green[PREDICTED POPULATION (LINEAR):  ] " + pred_pop_lin)

    else:
        st.write('PLEASE ENTER A VALID YEAR')

# Побудова графіка
with col2:
    if year.isnumeric():
        st.write('#### :green[' + option.upper() + "'S POPULATION]")

        selected_models = st.multiselect(
            'Select Models to Display on the Graph',
            options=['Polynomial Regression', 'Linear Regression', 'Historical Data'],
            default=['Polynomial Regression', 'Linear Regression', 'Historical Data']
        )

        fig = go.Figure()

        # Історичні дані
        if 'Historical Data' in selected_models:
            fig.add_trace(go.Scatter(x=countries_pop['Year'], y=countries_pop[option],
                                     name="Historical Population", line=dict(color='green', width=2)))

        # Поліноміальна регресія
        if 'Polynomial Regression' in selected_models:
            fig.add_trace(go.Scatter(x=years_poly.flatten(), y=pred_poly,
                                     name="Polynomial Regression Prediction", line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=[target_year], y=[pred_poly[-1]],
                                     name=f'Polynomial Prediction for {target_year}',
                                     mode='markers', marker_symbol='star',
                                     marker=dict(size=12, color='red')))

        # Лінійна регресія
        if 'Linear Regression' in selected_models:
            fig.add_trace(go.Scatter(x=years_lin.flatten(), y=pred_lin,
                                     name="Linear Regression Prediction", line=dict(color='blue', dash='dot')))
            fig.add_trace(go.Scatter(x=[target_year], y=[pred_lin[-1]],
                                     name=f'Linear Prediction for {target_year}',
                                     mode='markers', marker_symbol='star',
                                     marker=dict(size=12, color='blue')))

        # Оновлення параметрів графіка
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=600,
            width=900,
            title={
                'text': f"{option.upper()} Population Prediction",
                'x': 0.5,
                'y': 0.9,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': 'black'}
            },
            xaxis=dict(
                title="Year",
                titlefont=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black'),
                showgrid=True,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title="Population",
                titlefont=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black'),
                showgrid=True,
                gridcolor='lightgrey'
            ),
            font=dict(size=14, color='black'),
            legend=dict(
                font=dict(size=12, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        )

        st.plotly_chart(fig)
        st.write('The above plot shows the historical population and predictions from selected models. The star markers represent predictions for the selected year.')
