# AgriSetu: Project Methodology

The following is the structured 6-step pipeline methodology utilized by the AgriSetu backend to transition from raw environmental data to direct B2B farmer profitability.

## 1. Data Collection & Preprocessing
*   **Climate & Soil Data:** Gather daily historical Kerala weather metrics (`kerala_weather_2023_2024_full.csv`) and convert them into an 18-month rolling "Climatology Memory" to bypass inaccurate long-term weather forecasting. Gather absolute soil compatibility constraints (e.g., Loamy, Clay, Sandy).
*   **Economic History:** Extract 5 years of historical Agmarknet wholesale prices (`data/New/_price.csv`). 
*   **Harvest Records:** Parse governmental total-yield datasets (`kerala_monthly_estimates.csv`) to track historical state-wide crop success.

## 2. Biological Suitability Modeling (AI Engine)
*   Build an **XGBoost AI Classifier** to predict the pure biological probability of a crop surviving its entire multi-month gestation period.
*   **Scrubbing the "Acreage Illusion":** The AI uses time-traversal logic (`relativedelta`) to calculate the exact month seeds were planted. It looks up the historical price on that precise day and explicitly uses it as a **Control Variable** during training. During live site predictions, the AI is fed a fake "neutral price" so it ignores economics and predicts *strictly based on weather biology*.
*   **Quantization:** Raw harvest tons are classified into `[0 (Less Suitable), 1 (Moderately Suitable), 2 (Highly Suitable)]` using dynamic 33rd and 66th percentile thresholds of historical success. 
*   **Ensemble Fusion:** The AI's brutal historical thresholds are blended against a perfectly smooth "Gaussian Physics Simulator" to ensure natural mathematical leeway around extreme edge-cases.

## 3. Price Forecasting & Risk Assessment
*   Employ Facebook's **Prophet AI** framework to execute structural time-series forecasting on future Agmarknet prices.
*   Use Fourier-seasonality mapping to predict identical market gluts (e.g., price crashes when 10,000 farmers all harvest Paddy in December).
*   Calculate a rigid `volatility_index`. High-volatility crops are instantly flagged with a "High Risk" warning for the farmer.

## 4. Profitability Calculation
*   Compute Absolute Profitability taking all physical limits into account.
*   `Revenue = (Expected Yield/Acre × Predicted Future Price)`.
*   `Total Costs = Initial Input Costs + Monthly Maintenance Cost + 30% Post-Harvest Logistical Margin`.
*   Calculate the **Annualized Return on Investment (ROI%)** based on how many harvest cycles the specific crop can compress into 12 months.

## 5. Recommendation Generation
*   **Hard Constraint Filtering:** Any crop that exceeds the farmer's actual Budget limit or Max Duration limit is instantly purged from the matrix.
*   **Intercropping Optimization:** The matrix dynamically pairs primary recommendations with complementary companions (e.g., planting Ginger beneath tall Papaya trees to double land efficiency).
*   **Ranking Matrix:** Generate the final ranked list sorted by `hybrid_score` (Best Crops $\rightarrow$ Medium Profit $\rightarrow$ High Risk/Low Profit).
*   *Display Format:* "For your region, Pineapple is Highly Suitable, and is expected to generate 45.2% Annualized ROI per acre."

## 6. User Output (via a B2B Website)
*   **The Visualization:** Serve the generated matrix via a Flask REST API to the B2B web frontend. Display interactive visual risk-trend graphs and comprehensive ROI breakdowns. Offer alternative crop pivots highlighting explicit biological pros, cons, and estimated market demands. 
*   **B2B Integration Pipeline:** Provide an opportunity for farmers to view the projected date and volume of their future harvest (e.g., 50 MT in July 2025) and immediately list that yield on the website, allowing them to sell direct-to-customer or secure advanced B2B future-contracts from commercial buyers while the Prophet-predicted price is optimal.
