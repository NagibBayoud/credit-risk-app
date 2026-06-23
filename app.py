import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # الخوارزمية الجديدة
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(page_title="نظام إدارة ومقارنة مخاطر الائتمان", layout="wide")
st.title("🏦 لوحة تحكم المقارنة والتنبؤ بالقروض المتعثرة (NPL Dashboard)")

# دالة ذكية لتحديد سبب التعثر بناءً على المؤشرات المالية للعميل
def analyze_default_reasons(row):
    reasons = []
    try:
        if float(row['نسبة السيولة']) < 1.5:
            reasons.append("انخفاض نسبة السيولة (< 1.5)")
        if float(row['إجمالي الالتزامات']) > float(row['إجمالي الأصول']) * 0.6:
            reasons.append("ارتفاع نسبة الالتزامات للأصول")
        if 'السجل الائتماني' in row and row['السجل الائتماني'] in ['ضعيف', 'متوسط']:
            reasons.append(f"ضعف السجل الائتماني ({row['السجل الائتماني']})")
        if float(row['صافي الربح']) < float(row['الدخل السنوي']) * 0.1:
            reasons.append("ضآلة أو انخفاض صافي الربح")
    except:
        pass
    
    if not reasons:
        return "مخاطر ائتمانية عامة"
    return " | ".join(reasons)

# --- الخطوة 1: استيراد البيانات ---
st.sidebar.header("1. رفع البيانات")
uploaded_file = st.sidebar.file_uploader("قم برفع ملف البيانات (CSV, Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # قراءة البيانات
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
        
    st.subheader("📊 نظرة عامة على البيانات المستوردة")
    st.dataframe(df_raw.head(5), use_container_width=True)

    # --- الخطوة 2: إعدادات النموذج في القائمة الجانبية ---
    st.sidebar.header("2. إعدادات التحليل")
    target_col = st.sidebar.selectbox("عمود حالة القرض (المستهدف):", df_raw.columns, index=df_raw.columns.get_loc("حالة القرض") if "حالة القرض" in df_raw.columns else 0)
    id_col = st.sidebar.selectbox("عمود معرف/رقم العميل:", df_raw.columns, index=df_raw.columns.get_loc("رقم السجل") if "رقم السجل" in df_raw.columns else 0)
    
    # اختيار متعدد للخوارزميات
    selected_algorithms = st.sidebar.multiselect(
        "اختر الخوارزميات للمقارنة بينها:",
        ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost"],
        default=["Random Forest", "XGBoost"]
    )

    # تجهيز البيانات ومعالجتها للـ Machine Learning
    df = df_raw.copy().replace(r'^\s*$', np.nan, regex=True).dropna()
    
    # ترميز العمود المستهدف والاحتفاظ بالمخرجات النصية
    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col].astype(str))
    
    # ترميز باقي الأعمدة النصية
    le = LabelEncoder()
    for col in df.columns:
        if col != target_col and col != id_col:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = le.fit_transform(df[col].astype(str))

    # فصل البيانات للتمرين
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)

    # زر تشغيل التنبؤ والمقارنة
    if st.sidebar.button("تشغيل التنبؤ والمقارنة"):
        if not selected_algorithms:
            st.error("الرجاء اختيار خوارزمية واحدة على الأقل من القائمة الجانبية!")
        else:
            st.markdown("---")
            st.header("📈 لوحة المؤشرات ونتائج المقارنة (Dashboard)")
            
            # قاموس لحفظ نتائج الدقة والتنبؤات
            accuracy_results = {}
            results_table = pd.DataFrame({
                'رقم العميل': df_raw.loc[df.index, id_col],
                'حالة القرض الفعلية': df_raw.loc[df.index, target_col]
            })

            # تدريب الخوارزميات المختارة
            for algo in selected_algorithms:
                if algo == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif algo == "Logistic Regression":
                    model = LogisticRegression()
                elif algo == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif algo == "XGBoost":
                    model = XGBClassifier(random_state=42, eval_metric='logloss')
                
                # التدريب وحساب الدقة
                model.fit(X_train_scaled, y_train)
                preds_test = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds_test)
                accuracy_results[algo] = acc * 100
                
                # التنبؤ على كامل الجدول لعرضه أمام اسم العميل
                full_preds = model.predict(X_full_scaled)
                results_table[f'تنبؤ ({algo})'] = target_le.inverse_transform(full_preds)

            # إضافة أسباب التعثر ديناميكياً للعملاء المتعثرين فعلياً أو بالتنبؤ
            results_table['أسباب التعثر المحتملة'] = df_raw.loc[df.index].apply(
                lambda row: analyze_default_reasons(row) if row[target_col] == 'متعثر' else "وضع مالي مستقر / لا يوجد تعثر", axis=1
            )

            # --- عرض قسم المؤشرات المخططات البيانية ---
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.subheader("🎯 مقارنة دقة الخوارزميات المختارة")
                acc_df = pd.DataFrame(list(accuracy_results.items()), columns=['الخوارزمية', 'نسبة الدقة (%)'])
                fig_acc = px.bar(acc_df, x='الخوارزمية', y='نسبة الدقة (%)', text_auto='.2f', color='الخوارزمية', color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_acc, use_container_width=True)
                
            with col_m2:
                st.subheader("📊 توزيع حالات القروض (البيانات الحالية)")
                fig_pie = px.pie(df_raw, names=target_col, hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- عرض المخطط الثالث: المؤشرات المسببة للتعثر ---
            st.subheader("🛠️ المؤشرات الأكثر تأثيراً في اتخاذ القرار (مصفوفة الارتباط)")
            st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True, aspect="auto", color_continuous_scale='RdBu'), use_container_width=True)

            # --- عرض الجدول التفصيلي للعملاء مع أسباب التعثر ---
            st.markdown("---")
            st.header("📋 التقرير التفصيلي للعملاء والتنبؤات")
            
            # خانة بحث سريعة لفلترة عميل محدد
            search_customer = st.text_input("🔍 ابحث عن عميل محدد بواسطة (رقم/اسم العميل):")
            if search_customer:
                results_table = results_table[results_table['رقم العميل'].astype(str).str.contains(search_customer)]
                
            st.dataframe(results_table, use_container_width=True, hide_index=True)
            
else:
    st.info("💡 يرجى رفع ملف البيانات من القائمة الجانبية، ثم اختيار الخوارزميات والضغط على 'تشغيل التنبؤ والمقارنة'.")