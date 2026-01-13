import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset berdasarkan soal UAS
data = {
    'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Rainy', 'Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Rainy', 
                'Sunny', 'Cloudy', 'Cloudy', 'Rainy', 'Cloudy', 'Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Cloudy'],
    'Temp': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 
             'Medium', 'Medium', 'High', 'High', 'Medium', 'Medium', 'Low', 'Low', 'High', 'Medium'],
    'Wind': ['Weak', 'Weak', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Weak', 
             'Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong', 'Strong', 'Strong', 'Weak', 'Weak'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
             'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

def calculate_entropy(data_label):
    values, counts = np.unique(data_label, return_counts=True)
    entropy = 0
    for i in range(len(values)):
        prob = counts[i] / np.sum(counts)
        entropy -= prob * np.log2(prob)
    return entropy

def calculate_gain(data, attribute, target_name):
    total_entropy = calculate_entropy(data[target_name])
    values, counts = np.unique(data[attribute], return_counts=True)
    
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[attribute] == values[i]]
        prob = counts[i] / np.sum(counts)
        weighted_entropy += prob * calculate_entropy(subset[target_name])
    
    gain = total_entropy - weighted_entropy
    return gain

# --- PROSES DI TERMINAL ---
print("=== PERHITUNGAN ALGORITMA ID3 ===")
entropy_total = calculate_entropy(df['Play'])
print(f"1. Entropy Total S [20 Data]: {entropy_total:.4f}")

print("\n2. Menghitung Information Gain per Atribut:")
attributes = ['Weather', 'Temp', 'Wind']
gains = {}
for attr in attributes:
    gain = calculate_gain(df, attr, 'Play')
    gains[attr] = gain
    print(f"   - Gain(S, {attr}): {gain:.4f}")

root_node = max(gains, key=gains.get)
print(f"\n3. Root Node Terpilih: {root_node} (karena Gain tertinggi)")

print("\n4. Analisis Cabang pada Weather:")
for val in df['Weather'].unique():
    subset = df[df['Weather'] == val]['Play']
    ent = calculate_entropy(subset)
    print(f"   - Cabang {val}: Entropy = {ent:.4f} ({'Leaf Node' if ent == 0 else 'Perlu Split Lagi'})")

# --- VISUALISASI MATPLOTLIB ---
def draw_tree():
    fig, ax = plt.subplots(figsize=(10, 7))
    node_style = dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#01579b", lw=2)
    leaf_style = dict(boxstyle="round,pad=0.5", fc="#c8e6c9", ec="#2e7d32", lw=2)
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#546e7a")

    # Root
    ax.annotate(f"[{root_node}]", xy=(0.5, 0.9), ha='center', bbox=node_style, fontsize=12)

    # Cabang Cloudy (Leaf)
    ax.annotate("", xy=(0.2, 0.7), xytext=(0.45, 0.85), arrowprops=arrow_props)
    ax.text(0.28, 0.78, "Cloudy", rotation=25)
    ax.annotate("YES", xy=(0.2, 0.65), ha='center', bbox=leaf_style, fontsize=12, fontweight='bold')

    # Cabang Sunny (Split - Berdasarkan data Anda)
    ax.annotate("", xy=(0.5, 0.7), xytext=(0.5, 0.85), arrowprops=arrow_props)
    ax.text(0.52, 0.78, "Sunny")
    ax.annotate("NO", xy=(0.5, 0.65), ha='center', bbox=leaf_style, fontsize=12, fontweight='bold')

    # Cabang Rainy (Split ke Temperature)
    ax.annotate("", xy=(0.8, 0.7), xytext=(0.55, 0.85), arrowprops=arrow_props)
    ax.text(0.7, 0.78, "Rainy", rotation=-25)
    ax.annotate("[Temp]", xy=(0.8, 0.65), ha='center', bbox=node_style, fontsize=11)

    # Sub-cabang Rainy
    # Low -> Yes
    ax.annotate("", xy=(0.7, 0.45), xytext=(0.78, 0.6), arrowprops=arrow_props)
    ax.annotate("YES", xy=(0.7, 0.4), ha='center', bbox=leaf_style)
    ax.text(0.72, 0.52, "Low", fontsize=9)
    
    # High -> No
    ax.annotate("", xy=(0.9, 0.45), xytext=(0.82, 0.6), arrowprops=arrow_props)
    ax.annotate("NO", xy=(0.9, 0.4), ha='center', bbox=leaf_style)
    ax.text(0.85, 0.52, "High", fontsize=9)

    ax.set_axis_off()
    plt.title("Visualisasi Pohon Keputusan (Hasil ID3)", fontsize=14, pad=20)
    plt.show()

draw_tree()