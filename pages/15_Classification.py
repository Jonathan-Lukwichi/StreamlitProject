import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
from pathlib import Path

# Import custom modules
from app_core.state.session import init_state
from app_core.ui.theme import apply_css
from app_core.ui.theme import (
    PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    DANGER_COLOR, TEXT_COLOR, SUBTLE_TEXT, BODY_TEXT,
)
from app_core.ui.components import header
from app_core.ui.sidebar_brand import inject_sidebar_style, render_sidebar_brand

# Initialize session state
init_state()

# Page config
st.set_page_config(
    page_title="Classification - HealthForecast AI",
    page_icon="🎯",
    layout="wide",
)

apply_css()
inject_sidebar_style()
render_sidebar_brand()

# Fluorescent effects
st.markdown("""
<style>
/* ========================================
   FLUORESCENT EFFECTS FOR CLASSIFICATION
   ======================================== */

@keyframes float-orb {
    0%, 100% {
        transform: translate(0, 0) scale(1);
        opacity: 0.25;
    }
    50% {
        transform: translate(30px, -30px) scale(1.05);
        opacity: 0.35;
    }
}

.fluorescent-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(70px);
}

.orb-1 {
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 70%);
    top: 15%;
    right: 20%;
    animation: float-orb 25s ease-in-out infinite;
}

.orb-2 {
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.2), transparent 70%);
    bottom: 20%;
    left: 15%;
    animation: float-orb 30s ease-in-out infinite;
    animation-delay: 5s;
}

@keyframes sparkle {
    0%, 100% {
        opacity: 0;
        transform: scale(0);
    }
    50% {
        opacity: 0.6;
        transform: scale(1);
    }
}

.sparkle {
    position: fixed;
    width: 3px;
    height: 3px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.8), rgba(59, 130, 246, 0.3));
    border-radius: 50%;
    pointer-events: none;
    z-index: 2;
    animation: sparkle 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

.sparkle-1 { top: 25%; left: 35%; animation-delay: 0s; }
.sparkle-2 { top: 65%; left: 70%; animation-delay: 1s; }
.sparkle-3 { top: 45%; left: 15%; animation-delay: 2s; }

@media (max-width: 768px) {
    .fluorescent-orb {
        width: 200px !important;
        height: 200px !important;
        filter: blur(50px);
    }
    .sparkle {
        display: none;
    }
}
</style>

<!-- Fluorescent Floating Orbs -->
<div class="fluorescent-orb orb-1"></div>
<div class="fluorescent-orb orb-2"></div>

<!-- Sparkle Particles -->
<div class="sparkle sparkle-1"></div>
<div class="sparkle sparkle-2"></div>
<div class="sparkle sparkle-3"></div>
""", unsafe_allow_html=True)

# Premium Hero Header
st.markdown(
    f"""
    <div class='hf-feature-card' style='text-align: center; margin-bottom: 2rem;'>
      <div class='hf-feature-icon' style='margin: 0 auto 1.5rem auto;'>🎯</div>
      <h1 class='hf-feature-title' style='font-size: 2.5rem; margin-bottom: 1rem;'>Patient Arrival Reason Classification</h1>
      <p class='hf-feature-description' style='font-size: 1.125rem; max-width: 700px; margin: 0 auto;'>
        Predict and analyze reasons for patient arrivals using advanced machine learning classification models with comprehensive feature analysis
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for classification
if "classification_data" not in st.session_state:
    st.session_state.classification_data = None
if "classification_model" not in st.session_state:
    st.session_state.classification_model = None
if "classification_results" not in st.session_state:
    st.session_state.classification_results = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "feature_scaler" not in st.session_state:
    st.session_state.feature_scaler = None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Upload",
    "🤖 Model Training",
    "📈 Results & Metrics",
    "🔮 Prediction"
])

# ============================================================================
# TAB 1: Data Upload
# ============================================================================
with tab1:
    header("📊", "Data Upload", "Upload dataset with patient arrival reasons")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Upload Classification Dataset")
        st.info("Dataset should contain features and a target column with reason categories (e.g., Emergency, Scheduled Appointment, Walk-in, etc.)")

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="classification_upload"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.classification_data = df

                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

                st.markdown("#### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("#### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    with col2:
        if st.session_state.classification_data is not None:
            df = st.session_state.classification_data

            st.markdown("### Dataset Statistics")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", len(df.columns))
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

            # Show unique values for categorical columns
            st.markdown("### Categorical Columns")
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols[:5]:  # Show first 5
                unique_count = df[col].nunique()
                st.metric(col, unique_count, delta="unique values")

# ============================================================================
# TAB 2: Model Training
# ============================================================================
with tab2:
    header("🤖", "Model Training", "Train classification models to predict arrival reasons")

    if st.session_state.classification_data is None:
        st.warning("⚠️ Please upload data in the Data Upload tab first.")
    else:
        df = st.session_state.classification_data.copy()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Configuration")

            # Select target column
            target_col = st.selectbox(
                "Select Target Column (Reason)",
                options=df.columns.tolist(),
                help="Column containing the reason categories"
            )

            # Select feature columns
            available_features = [col for col in df.columns if col != target_col]
            feature_cols = st.multiselect(
                "Select Feature Columns",
                options=available_features,
                default=available_features[:min(10, len(available_features))],
                help="Columns to use as predictors"
            )

            # Model selection
            model_type = st.selectbox(
                "Select Classification Model",
                options=[
                    "Random Forest",
                    "Gradient Boosting",
                    "Logistic Regression"
                ]
            )

            # Train/test split
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            ) / 100

            random_state = st.number_input(
                "Random State (for reproducibility)",
                min_value=0,
                max_value=1000,
                value=42
            )

        with col2:
            st.markdown("### Model Hyperparameters")

            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                max_depth = st.slider("Max Depth", 3, 50, 10, 1)
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)

            elif model_type == "Gradient Boosting":
                n_estimators = st.slider("Number of Estimators", 10, 500, 100, 10)
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
                max_depth = st.slider("Max Depth", 3, 20, 3, 1)

            elif model_type == "Logistic Regression":
                C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
                max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)

        st.markdown("---")

        # Train button
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if not feature_cols:
                st.error("Please select at least one feature column!")
            else:
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()

                        # Handle missing values
                        X = X.fillna(X.mean(numeric_only=True))
                        X = X.fillna(method='ffill').fillna(method='bfill')

                        # Encode categorical features
                        for col in X.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))

                        # Encode target
                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y)
                        st.session_state.label_encoder = label_encoder

                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                        )

                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        st.session_state.feature_scaler = scaler

                        # Train model
                        if model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state,
                                n_jobs=-1
                            )
                        elif model_type == "Gradient Boosting":
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                random_state=random_state
                            )
                        else:  # Logistic Regression
                            model = LogisticRegression(
                                C=C,
                                max_iter=max_iter,
                                random_state=random_state,
                                multi_class='multinomial'
                            )

                        model.fit(X_train_scaled, y_train)

                        # Make predictions
                        y_train_pred = model.predict(X_train_scaled)
                        y_test_pred = model.predict(X_test_scaled)

                        # Calculate metrics
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        test_accuracy = accuracy_score(y_test, y_test_pred)

                        # Multi-class metrics (weighted average)
                        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                        # Confusion matrix
                        cm = confusion_matrix(y_test, y_test_pred)

                        # Classification report
                        report = classification_report(
                            y_test, y_test_pred,
                            target_names=label_encoder.classes_,
                            output_dict=True,
                            zero_division=0
                        )

                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                        else:
                            feature_importance = None

                        # Store results
                        st.session_state.classification_model = model
                        st.session_state.classification_results = {
                            'model_type': model_type,
                            'train_accuracy': train_accuracy,
                            'test_accuracy': test_accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'confusion_matrix': cm,
                            'classification_report': report,
                            'feature_importance': feature_importance,
                            'feature_cols': feature_cols,
                            'target_col': target_col,
                            'classes': label_encoder.classes_.tolist(),
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_test_pred': y_test_pred,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        st.success(f"✅ Model trained successfully! Test Accuracy: {test_accuracy:.2%}")
                        st.balloons()

                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# ============================================================================
# TAB 3: Results & Metrics
# ============================================================================
with tab3:
    header("📈", "Results & Metrics", "Evaluate classification model performance")

    if st.session_state.classification_results is None:
        st.warning("⚠️ Please train a model first in the Model Training tab.")
    else:
        results = st.session_state.classification_results

        # Metrics cards
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{results['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{results['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{results['f1_score']:.2%}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = results['confusion_matrix']
            classes = results['classes']

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))

            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # Classification Report
            st.markdown("### Classification Report by Class")

            report_df = pd.DataFrame(results['classification_report']).T
            report_df = report_df[report_df.index.isin(classes)]
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
            report_df = report_df.round(3)

            st.dataframe(report_df, use_container_width=True)

            # Bar chart of F1 scores by class
            fig_f1 = px.bar(
                x=report_df.index,
                y=report_df['f1-score'],
                labels={'x': 'Class', 'y': 'F1 Score'},
                title='F1 Score by Class',
                color=report_df['f1-score'],
                color_continuous_scale='Viridis'
            )

            fig_f1.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )

            st.plotly_chart(fig_f1, use_container_width=True)

        # Feature Importance
        if results['feature_importance'] is not None:
            st.markdown("---")
            st.markdown("### Feature Importance")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_imp = px.bar(
                    results['feature_importance'].head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='Importance',
                    color_continuous_scale='Plasma'
                )

                fig_imp.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig_imp, use_container_width=True)

            with col2:
                st.markdown("#### Top 10 Features")
                st.dataframe(
                    results['feature_importance'].head(10),
                    use_container_width=True,
                    hide_index=True
                )

        # Model info
        st.markdown("---")
        st.markdown("### Model Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Model Type:** {results['model_type']}")
        with col2:
            st.info(f"**Number of Classes:** {len(results['classes'])}")
        with col3:
            st.info(f"**Trained:** {results['timestamp']}")

        # Download model
        st.markdown("---")
        if st.button("💾 Save Model", use_container_width=True):
            try:
                # Create artifacts directory
                artifacts_dir = Path("pipeline_artifacts/classification")
                artifacts_dir.mkdir(parents=True, exist_ok=True)

                # Save model
                model_path = artifacts_dir / f"model_{results['model_type'].replace(' ', '_')}.pkl"
                joblib.dump(st.session_state.classification_model, model_path)

                # Save label encoder
                encoder_path = artifacts_dir / "label_encoder.pkl"
                joblib.dump(st.session_state.label_encoder, encoder_path)

                # Save scaler
                scaler_path = artifacts_dir / "scaler.pkl"
                joblib.dump(st.session_state.feature_scaler, scaler_path)

                st.success(f"✅ Model saved to {artifacts_dir}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

# ============================================================================
# TAB 4: Prediction
# ============================================================================
with tab4:
    header("🔮", "Prediction", "Predict arrival reason for new patients")

    if st.session_state.classification_model is None:
        st.warning("⚠️ Please train a model first in the Model Training tab.")
    else:
        results = st.session_state.classification_results
        model = st.session_state.classification_model

        st.markdown("### Make Predictions")

        prediction_mode = st.radio(
            "Prediction Mode",
            options=["Single Prediction", "Batch Prediction"],
            horizontal=True
        )

        if prediction_mode == "Single Prediction":
            st.markdown("#### Enter Feature Values")

            # Create input fields for each feature
            feature_values = {}
            cols = st.columns(3)

            for idx, feature in enumerate(results['feature_cols']):
                with cols[idx % 3]:
                    feature_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.2f"
                    )

            if st.button("🎯 Predict", type="primary", use_container_width=True):
                try:
                    # Prepare input
                    input_df = pd.DataFrame([feature_values])
                    input_scaled = st.session_state.feature_scaler.transform(input_df)

                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]

                    # Decode prediction
                    predicted_class = st.session_state.label_encoder.inverse_transform([prediction])[0]

                    # Display result
                    st.markdown("---")
                    st.success(f"### 🎯 Predicted Reason: **{predicted_class}**")

                    # Probability distribution
                    st.markdown("#### Prediction Probabilities")
                    proba_df = pd.DataFrame({
                        'Reason': results['classes'],
                        'Probability': prediction_proba
                    }).sort_values('Probability', ascending=False)

                    fig_proba = px.bar(
                        proba_df,
                        x='Reason',
                        y='Probability',
                        title='Prediction Confidence by Class',
                        color='Probability',
                        color_continuous_scale='Turbo'
                    )

                    fig_proba.update_layout(
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )

                    st.plotly_chart(fig_proba, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

        else:  # Batch Prediction
            st.markdown("#### Upload Data for Batch Prediction")

            batch_file = st.file_uploader(
                "Choose CSV file with same features",
                type=["csv"],
                key="batch_prediction"
            )

            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)

                    # Validate features
                    missing_features = set(results['feature_cols']) - set(batch_df.columns)
                    if missing_features:
                        st.error(f"Missing features: {missing_features}")
                    else:
                        X_batch = batch_df[results['feature_cols']].copy()

                        # Handle missing values
                        X_batch = X_batch.fillna(X_batch.mean(numeric_only=True))
                        X_batch = X_batch.fillna(method='ffill').fillna(method='bfill')

                        # Encode categorical features
                        for col in X_batch.select_dtypes(include=['object']).columns:
                            le = LabelEncoder()
                            X_batch[col] = le.fit_transform(X_batch[col].astype(str))

                        # Scale and predict
                        X_batch_scaled = st.session_state.feature_scaler.transform(X_batch)
                        predictions = model.predict(X_batch_scaled)
                        predictions_proba = model.predict_proba(X_batch_scaled)

                        # Decode predictions
                        predicted_classes = st.session_state.label_encoder.inverse_transform(predictions)

                        # Add predictions to dataframe
                        batch_df['Predicted_Reason'] = predicted_classes
                        batch_df['Prediction_Confidence'] = predictions_proba.max(axis=1)

                        st.success(f"✅ Predictions made for {len(batch_df)} records!")

                        # Display results
                        st.dataframe(batch_df, use_container_width=True)

                        # Distribution of predictions
                        st.markdown("#### Prediction Distribution")
                        pred_dist = pd.DataFrame(predicted_classes, columns=['Predicted_Reason'])
                        fig_dist = px.pie(
                            pred_dist,
                            names='Predicted_Reason',
                            title='Distribution of Predicted Reasons'
                        )

                        fig_dist.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )

                        st.plotly_chart(fig_dist, use_container_width=True)

                        # Download predictions
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error processing batch predictions: {str(e)}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "HealthForecast AI - Classification Module | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
