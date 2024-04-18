import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Load breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
feature_names = breast_cancer.feature_names

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
st.write("Breast Cancer Database: ")
st.write(df)

# Streamlit web app
def main():
    st.title("Breast Cancer Detection")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Sidebar with user input
    st.sidebar.header("User Input")
    selected_feature = st.sidebar.selectbox("Select Feature", feature_names)

    VIDEO_URL = "https://youtu.be/Ig1-n4X8pCY?si=Imi41_o1KKrB5VbC"
    st.video(VIDEO_URL)

    # Display histogram of selected feature
    st.header("Histogram of Selected Feature")
    plt.figure(figsize=(8, 6))
    plt.hist(df[selected_feature], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(selected_feature)
    plt.ylabel("Frequency")
    st.pyplot()

    # Display scatter plot of two features
    st.header("Scatter Plot of Two Features")
    selected_feature2 = st.sidebar.selectbox("Select Another Feature", feature_names)
    plt.figure(figsize=(8, 6))
    plt.scatter(df[selected_feature], df[selected_feature2], c=df['target'], cmap='viridis')
    plt.xlabel(selected_feature)
    plt.ylabel(selected_feature2)
    st.pyplot()

    # Display bar chart of class distribution
    st.header("Bar Chart of Class Distribution")
    class_counts = df['target'].value_counts()
    labels = ['Benign', 'Malignant']
    plt.figure(figsize=(6, 6))
    plt.bar(labels, class_counts, color=['blue', 'red'])
    plt.xlabel("Class")
    plt.ylabel("Count")
    st.pyplot()

    # Display pie chart of class distribution
    st.header("Pie Chart of Class Distribution")
    class_counts = df['target'].value_counts()
    labels = ['Benign', 'Malignant']
    st.write(class_counts)
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    st.pyplot()

    st.write("")

    # Data
    data = {
        'Place Name': [
            'Guru Kirpa, Best Rehab Center in Punjab',
            'Ruby Hall Clinic Wanowrie, Maharashtra, India',
            'Fortis Memorial Research Institute, Gurgaon, India',
            'Ahmedabad Civil Hospital, Ahmedabad, Gujarat, India',
            'Aakrithi Hospital, Vijayawada, Andhra Pradesh, India',
            'VOC Port Trust Hospital, Muttayyapuram, Tamil Nadu , India',
            'Vihar Hospital, Anand, Gujarat, India',
            'Advanced Centre for Eyes, Kitchlu Nagar, Ludhiana, Punjab, India',
            'Delhi Heart Hospital, Jagriti Enclave, Anand Vihar, Delhi, India',
            'Nighasan Hospital, Nighasan, Uttar Pradesh, India',
            'Apple Hospital, Surat, Gujarat, India',
            'Primary Health Centre, Gejjalagere, Karnataka, India',
            '32 Smile Stone Dental Clinic, New Delhi, Delhi, India',
            'Veterinary Polyclinic, Hoshiarpur, Punjab, India',
            'Hashmika Child Clinic, Visakhapatnam, Andhra Pradesh, India',
            'Padmini Nursing Home, Chetpet, Chennai, Tamil Nadu, India',
            'Subham Diagnostic & Polyclinic, Rajhati, West Bengal, India',
            'Smile Art Dental Clinic, Ravet, Pimpri-Chinchwad, Maharashtra, India',
            'Asilo Hospital, Mapusa, Goa, India',
            'General Hospital, Jangipara, Hooghly, West Bangali, India',
            'Western India Institute Of Neurosciences, Nagala Park, Kolhapur, Maharashtra, India',
            'MGM Hospital and Research Center, Katni, Madhya Pradesh, India',
            'Primary Health Care Center, Pataka, Athmallik, Odisha, India',
            'Jyotirmayee Medicine Store, Pataka, Athmallik, Odisha, India',
            'Androbest Andrology & Urology Center, Sai Nagar, LB Nagar, Hyderabad, Telangana, India',
            'ADORN Cosmetic Clinic, Ahmedabad, Gujarat, India',
            'Vignesh Hospital, Porur, Ramapuram, Chennai, Tamil Nadu, India',
            'Chennai Jayanth Acupuncture Hospital, Anna Nagar, Chennai, Tamil Nadu, India',
            'Srinivas Priya Hospital Pvt Ltd, Patel Road, Perambur, Chennai, India',
            'RELAX Hospital, Cuttack, Orrisa, Odisha, India',
            'Governmental Hospital, Bachannapet, Telangana, India',
            'Governmental Hospital of Thalaivasal, Thalaivasal, Tamil Nadu, India',
            'Dr deepa shama\'s DEEP Hospital, Hathras, Uttar Pradesh, India',
            'Aark Foundation, Donje Phata, Pune, Maharashtra, India',
            'Sant Blood Bank, Jhansi, Uttar Pradesh, India',
            'Riddhi Siddhi CHS, Borivali West, Mumbai, Maharashtra, India',
            'MAURYA Eye Care Center, Manikpur, Uttar Pradesh, India',
            'Dental Panacea, Faridabad, Hayrana, India',
            'Srirangam Government Hospital, Tiruchirappalli, Tamil Nadu, India',
            'Bairabi hospital, Bairabi, Mizoram, India',
            'Khuangpuilam Clinic, Kolasib, Mizoram, India',
            'Nityanand Hospital, Katraj, Pune, Maharashtra, India',
            'Hojai Civil Hospital, Hojai, Assam, India',
            'Apollo BSR Hospital, Bhilai Nagar, Chhattisgarh, India',
            'Usha Vision Care, Srirampura, Bengaluru, Karnataka, India',
            'Keshav Madhav Blood Bank, Bareilly, Uttar Pradesh, India',
            'Mukta Dental Clinic, Shahid Bhagat Singh Nagar, Rajasthan, India',
            'MJM Hospital, Shivajinagar, Pune, Maharashtra, India',
            'Sarthak Manav Kusthashram, Jhotwara, Jaipur, Rajasthan, India',
            'Janta Clinic, Sector 3, Jaipur, Rajasthan, India',
            'Pashu Hospital Maheshwar, Maheshwar, Madhya Pradesh, India',
            'Sagar Hospital, KumaraSwamy layout, Bangalore, Karnataka, India',
            'Dr.Shruthi and Dr.Rajesh Patil, Rajatagiri, Dharwad, Karnataka, India',
            'UHP District General Hospital, Armavti, Maharashtra, India',
            'MGM Hospital, CBD Belapur, Mumbai, Maharashtra, India',
            'Pramathana Dental Care, Ideal Homes TWP, Bengaluru, Karnataka, India',
            'Joshi Hospital, Dabhade, Pune, Maharashtra, India',
            'Rural Hospital Solankur, Solankur, Maharashtra, India',
            '32 Gems Dental Care, Dosarka, Punjab, India',
            'Subham hospital, Mendarda, Sardarbag, Junagadh, Gujarat, India',
            'Hiranandani Hospital, Thane West, Mumbai, Maharashtra',
            'Sadar Hospital, Jamshedpur, India',
            'Fairbank James Friendship memorial Hospital, Ahmednagar, Maharashtra, India',
            'Medicare Skin & Cosmetic Clinic, Jayanagar, Bangalore, India',
            'Banglore Hospital, Bengaluru, Karnataka, India',
            'Indus Hospital, Sector 60, Punjab, India',
            'Tiruvalla Medical Mission Hospital, Kerala, India'
        ],
        'Latitude': [
            31.096134, 18.485870, 28.456789, 23.053967, 16.511965,
            8.749402, 22.554609, 30.912411, 28.653229, 28.231674,
            21.182947, 12.571047, 28.575552, 31.524620, 17.733288,
            13.072790, 22.674788, 18.643318, 15.589379, 22.742229,
            16.709822, 23.830975, 20.651484, 20.650694, 17.357861,
            23.025570, 13.030947, 13.095658, 13.109593, 20.457838,
            17.786711, 11.578299, 27.597265, 18.399286, 25.458599,
            19.227650, 25.766827, 28.386002, 10.857012, 24.184324,
            24.209656, 18.457527, 26.001802, 21.216276, 12.996090,
            28.367180, 24.558990, 18.524338, 26.940351, 26.889633,
            22.179298, 12.907950, 15.437003, 20.933424, 19.025806,
            12.923236, 18.970409, 16.413506, 31.697235, 21.323082,
            19.252562, 22.758537, 19.092508, 12.906529, 13.006752,
            30.705317, 9.393924
        ],
        'Longitude': [
            75.778770, 73.905853, 77.072472, 72.603844, 80.633163,
            78.168137, 72.967361, 75.819412, 77.308601, 80.862534,
            72.831581, 77.001183, 77.262192, 75.902008, 83.275429,
            80.234421, 87.827484, 73.756042, 73.816574, 88.051460,
            74.227463, 80.407120, 84.629814, 84.631775, 78.557442,
            72.527458, 80.171585, 80.206116, 80.246666, 85.871536,
            79.026970, 78.753654, 78.045441, 73.769058, 78.615517,
            72.840012, 81.414467, 77.307678, 78.691162, 92.533638,
            92.679642, 73.867668, 92.848373, 81.323608, 77.569672,
            79.430153, 73.722801, 73.843887, 75.769493, 75.839554,
            75.586754, 77.565063, 75.015060, 77.761139, 73.041550,
            77.518456, 76.753838, 74.050575, 75.788933, 70.441826,
            72.980057, 86.201302, 74.749596, 77.585831, 77.561737,
            76.725052, 76.578423
        ]
    }
    # Create DataFrame
    map_df = pd.DataFrame(data)
    map_df.rename(columns={'Latitude': 'latitude'}, inplace=True)
    # Rename Longitude column to lowercase 'longitude'
    map_df.rename(columns={'Longitude': 'longitude'}, inplace=True)
    # Show map
    map_zoom = 3
    # Add a heading above the map
    st.subheader("Breast Cancer Treatment Centers Across India")
    st.map(map_df, use_container_width=True, zoom = map_zoom)
st.balloons()
if __name__ == "__main__":
    main()
