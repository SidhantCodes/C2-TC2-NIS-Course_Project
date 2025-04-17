import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from fpdf import FPDF
import torch
import torch.nn as nn
import joblib
import numpy as np

def get_top_attack_categories(df, n=5):
    attacks = df[df['label'] != 'BenignTraffic']['label'].value_counts().head(n)
    result = []
    for attack, count in attacks.items():
        result.append(f"{attack}: {count:,} flows")
    return '\n  '.join(result)

def get_example_violations(df, rule, n=3):
    examples = df[df['rejection_reasons'].str.contains(rule)].head(n)
    if examples.empty:
        return "No examples found"
    result = []
    for _, row in examples.iterrows():
        features = []
        if 'HTTP' in row.index and row['HTTP'] > 0:
            features.append("HTTP")
        if 'ICMP' in row.index and row['ICMP'] > 0:
            features.append("ICMP")
        if 'TCP' in row.index and row['TCP'] > 0:
            features.append("TCP")
        if 'UDP' in row.index and row['UDP'] > 0:
            features.append("UDP")
        if 'syn_flag_number' in row.index and row['syn_flag_number'] > 0:
            features.append("SYN flags")
        if 'fin_flag_number' in row.index and row['fin_flag_number'] > 0:
            features.append("FIN flags")
        protocol = ', '.join(features) or f"Protocol {row.get('Protocol Type', 'Unknown')}"
        rate = row.get('Rate', 'N/A')
        rate_str = f"{rate:.1f}" if isinstance(rate, (int, float)) else rate
        duration = row.get('Duration', 'N/A')
        duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else duration
        result.append(f"Flow with {protocol}, Rate={rate_str}, Duration={duration_str}")
    return '\n      '.join(result)

def severity_score(severity):
    scores = {
        'Critical': 4,
        'High': 3,
        'Medium': 2,
        'Low': 1
    }
    return scores.get(severity, 0)

def generate_detailed_security_report(df, approved_df, initial_count, rejection_log, security_rules, model_accuracy, uncertain_df):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rejected_count = initial_count - len(approved_df)
    rejected_percent = rejected_count / initial_count * 100
    critical_count = sum(1 for _, row in df.iterrows() if row['severity'] == 'Critical')
    high_count = sum(1 for _, row in df.iterrows() if row['severity'] == 'High')
    medium_count = sum(1 for _, row in df.iterrows() if row['severity'] == 'Medium')
    ddos_count = sum(1 for _, row in approved_df.iterrows() if 'DDoS' in str(row['label']))
    dos_count = sum(1 for _, row in approved_df.iterrows() if 'DoS' in str(row['label']) and 'DDoS' not in str(row['label']))
    recon_count = sum(1 for _, row in approved_df.iterrows() if 'Recon' in str(row['label']))
    injection_count = sum(1 for _, row in approved_df.iterrows() if any(x in str(row['label']) for x in ['Injection', 'XSS', 'SQL']))
    malware_count = sum(1 for _, row in approved_df.iterrows() if any(x in str(row['label']) for x in ['Malware', 'Backdoor', 'Mirai']))
    other_attacks = sum(1 for _, row in approved_df.iterrows() if row['label'] != 'BenignTraffic' and not any(
        x in str(row['label']) for x in ['DDoS', 'DoS', 'Recon', 'Injection', 'XSS', 'SQL', 'Malware', 'Backdoor', 'Mirai']
    ))
    benign_count = sum(1 for _, row in approved_df.iterrows() if row['label'] == 'BenignTraffic')
    report = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃                       ENTERPRISE NETWORK SECURITY REPORT                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────────────────────
Report Generated: {timestamp}
Total Network Flows Analyzed: {initial_count:,}
Flows Rejected by Security Controls: {rejected_count:,} ({rejected_percent:.1f}%)
Flows Processed for Deep Analysis: {len(approved_df):,} ({100-rejected_percent:.1f}%)

Critical Security Issues: {critical_count:,}
High Severity Issues: {high_count:,}
Medium Severity Issues: {medium_count:,}

SECURITY RULE ENFORCEMENT DETAILS
─────────────────────────────────────────────────────────────────────────────────────"""
    sorted_rules = sorted([(rule, count) for rule, count in rejection_log.items()], 
                         key=lambda x: x[1], reverse=True)
    for rule_name, count in sorted_rules:
        if count > 0 and rule_name in security_rules:
            rule_data = security_rules[rule_name]
            report += f"""

■ {rule_name} ({rule_data['severity']})
  Flows Affected: {count:,}
  Description: {rule_data['reason']}
  MITRE ATT&CK: {rule_data['mitre']}
  CVE Reference: {rule_data['cve']}
  
  Example Violations:
      {get_example_violations(df, rule_name)}"""
    report += f"""

MACHINE LEARNING MODEL PERFORMANCE
─────────────────────────────────────────────────────────────────────────────────────
Model Accuracy on Approved Traffic: {model_accuracy*100:.1f}%
Total Benign Flows Identified: {benign_count:,}
Total Attack Flows Detected: {len(approved_df) - benign_count:,}

Attack Type Distribution:
  DDoS Attacks: {ddos_count:,}
  DoS Attacks: {dos_count:,}
  Reconnaissance: {recon_count:,}
  Injection Attacks: {injection_count:,}
  Malware Traffic: {malware_count:,}
  Other Attack Types: {other_attacks:,}


Stage 2 Autoencoder Analysis:
  Uncertain Flows Sent to Autoencoder: {len(uncertain_df):,}
  Anomalies Detected by Autoencoder: {uncertain_df["stage2_anomaly"].sum():,}

Top Attack Categories:
  {get_top_attack_categories(approved_df, 5)}

TREND ANALYSIS
─────────────────────────────────────────────────────────────────────────────────────
Key Observations:
• {get_key_observation(df, rejection_log)}
• {get_protocol_observation(df)}
• {get_rate_observation(df)}

For detailed traffic analytics and a complete breakdown of all blocked traffic, 
refer to the exported 'security_processed_results.csv' file.
"""
    return report

def get_key_observation(df, rejection_log):
    if not rejection_log:
        return "No security rule violations detected"
    top_rule = max(rejection_log.items(), key=lambda x: x[1])[0]
    return f"Highest security rule violation: {top_rule} with {rejection_log[top_rule]:,} flows"

def get_protocol_observation(df):
    http_count = sum(1 for _, row in df.iterrows() if 'HTTP' in row and row['HTTP'] > 0)
    https_count = sum(1 for _, row in df.iterrows() if 'HTTPS' in row and row['HTTPS'] > 0)
    if http_count > https_count:
        return f"Insecure HTTP protocol usage ({http_count:,} flows) exceeds secure HTTPS ({https_count:,} flows)"
    else:
        return f"Secure HTTPS traffic ({https_count:,} flows) exceeds insecure HTTP ({http_count:,} flows)"

def get_rate_observation(df):
    high_rate = sum(1 for _, row in df.iterrows() if 'Rate' in row and row['Rate'] > 1000)
    return f"High-rate traffic flows detected: {high_rate:,} flows with rate >1000 pps"

SECURITY_RULES = {
    'Unsecured HTTP Traffic': {
        'detector': lambda df: df['HTTP'] > 0,
        'reason': "Clear-text HTTP traffic detected (NIST SC-8, SC-13 compliance)",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1071.001'
    },
    'Unencrypted Sensitive Service': {
        'detector': lambda df: (df['Telnet'] > 0) | (df['SMTP'] > 0),
        'reason': "Legacy protocols without encryption detected (NIST SC-8)",
        'severity': 'High',
        'cve': 'N/A',
        'mitre': 'T1190'
    },
    'Invalid Protocol Combinations': {
        'detector': lambda df: ((df['TCP'] > 0) & (df['UDP'] > 0)) | ((df['ICMP'] > 0) & ((df['HTTP'] > 0) | (df['HTTPS'] > 0))),
        'reason': "Illegal protocol encapsulation detected (possible tunneling)",
        'severity': 'High',
        'cve': 'N/A',
        'mitre': 'T1572'
    },
    'SYN Flood Pattern': {
        'detector': lambda df: (df['syn_flag_number'] > 3) & (df['ack_flag_number'] == 0) & (df['Rate'] > 1000),
        'reason': "TCP SYN flood pattern detected (OWASP Top 10 DoS)",
        'severity': 'Critical',
        'cve': 'CVE-2023-1017',
        'mitre': 'T1499.002'
    },
    'ICMP Flood Pattern': {
        'detector': lambda df: (df['ICMP'] > 0) & (df['Number'] > 100) & (df['Magnitude'] < 5),
        'reason': "ICMP flood with small payload detected",
        'severity': 'High',
        'cve': 'N/A',
        'mitre': 'T1499.001'
    },
    'RST/FIN Flood Pattern': {
        'detector': lambda df: (df['fin_flag_number'] > 0) & (df['rst_flag_number'] > 0) & (df['Rate'] > 100),
        'reason': "TCP RST/FIN flood pattern detected (CIS Control 13)",
        'severity': 'Critical',
        'cve': 'N/A',
        'mitre': 'T1499.002'
    },
    'Invalid Header Length': {
        'detector': lambda df: (df['Header_Length'] < 20) | (df['Header_Length'] > 1000),
        'reason': "Malformed packet headers detected (possible IDS evasion)",
        'severity': 'High',
        'cve': 'CVE-2021-44228',
        'mitre': 'T1562.003'
    },
    'Abnormal Flow Duration': {
        'detector': lambda df: df['flow_duration'] > 1e6,
        'reason': "Extremely long session duration (C2 channel indicator)",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1071.004'
    },
    'Short-lived TCP Connection': {
        'detector': lambda df: (df['TCP'] > 0) & (df['Duration'] < 0.01) & (df['Number'] > 10),
        'reason': "Rapid connection teardown pattern (scanning behavior)",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1046'
    },
    'Suspicious Data Transfer Rate': {
        'detector': lambda df: df['Rate'] > 10000,
        'reason': "Abnormally high data transfer rate (data exfiltration)",
        'severity': 'High',
        'cve': 'N/A',
        'mitre': 'T1048'
    },
    'Suspicious Payload Size': {
        'detector': lambda df: (df['Tot size'] > 1e6) | ((df['AVG'] < 10) & (df['Number'] > 50)),
        'reason': "Anomalous payload characteristics (exfiltration channel)",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1020'
    },
    'SQL Injection Pattern': {
        'detector': lambda df: (df['Weight'] > 90) & (df['HTTP'] > 0) & (df['urg_count'] > 0),
        'reason': "HTTP traffic with suspicious SQL injection patterns",
        'severity': 'Critical',
        'cve': 'CVE-2020-13379',
        'mitre': 'T1190'
    },
    'Command Injection Pattern': {
        'detector': lambda df: (df['HTTP'] > 0) & (df['Weight'] > 75) & (df['Std'] > 10),
        'reason': "HTTP traffic with command delimiter patterns",
        'severity': 'Critical',
        'cve': 'CVE-2021-42013',
        'mitre': 'T1059.004'
    },
    'XSS Attack Pattern': {
        'detector': lambda df: (df['HTTP'] > 0) & (df['Weight'] > 80) & (df['Tot size'] > 1000),
        'reason': "HTTP traffic with script injection indicators",
        'severity': 'High',
        'cve': 'CVE-2021-44228',
        'mitre': 'T1059.007'
    },
    'Network Scanning Pattern': {
        'detector': lambda df: (df['ICMP'] > 0) & (df['Duration'] < 0.1) & (df['Number'] > 5),
        'reason': "ICMP-based network scanning pattern detected",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1595.001'
    },
    'Port Scanning Pattern': {
        'detector': lambda df: (df['Rate'] > 100) & (df['Tot size'] < 100) & ((df['UDP'] > 0) | (df['TCP'] > 0)),
        'reason': "High-speed port scanning activity detected",
        'severity': 'Medium',
        'cve': 'N/A',
        'mitre': 'T1046'
    },
    'Covert Channel Pattern': {
        'detector': lambda df: (df['flow_duration'] > 1000) & (df['Rate'] < 10) & (df['Covariance'] > 5),
        'reason': "Long-duration low-rate covert channel detected",
        'severity': 'High',
        'cve': 'N/A',
        'mitre': 'T1071.001'
    }
}

try:
    with open('random_f.pkl', 'rb') as f:
        model = pickle.load(f)
    with open("label_enc_rf.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except FileNotFoundError as e:
    exit(1)
except Exception as e:
    exit(1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_input_dim = 26
autoencoder = Autoencoder(ae_input_dim).to(device)
try:
    autoencoder.load_state_dict(torch.load("stage2_qae_model.pth", map_location=device))
    ae_scaler = joblib.load("stage2_qae_scaler.pkl")
except FileNotFoundError as e:
    autoencoder = Autoencoder(ae_input_dim).to(device)
    ae_scaler = QuantileTransformer(output_distribution='uniform')
except Exception as e:
    autoencoder = Autoencoder(ae_input_dim).to(device)
    ae_scaler = QuantileTransformer(output_distribution='uniform')
autoencoder.eval()

selected_features = ['urg_count', 'Min', 'syn_flag_number', 'Duration', 'flow_duration',
                     'Protocol Type', 'rst_count', 'Variance', 'Header_Length', 'HTTP', 'UDP',
                     'TCP', 'Max', 'syn_count', 'Rate', 'Tot size', 'ICMP', 'psh_flag_number',
                     'Covariance', 'AVG', 'fin_flag_number', 'ack_flag_number', 'Tot sum',
                     'HTTPS', 'IAT', 'fin_count']

try:
    df = pd.read_csv('three.csv')
    initial_count = len(df)
except FileNotFoundError:
    exit(1)
except Exception as e:
    exit(1)

missing_features = [feature for feature in selected_features if feature not in df.columns]
if missing_features:
    for feature in missing_features:
        df[feature] = 0

rejection_log = defaultdict(int)
df['rejection_reasons'] = ''
df['security_risk_score'] = 0
df['severity'] = 'Unknown'

for rule_name, rule_data in SECURITY_RULES.items():
    detector = rule_data['detector']
    severity = rule_data['severity']
    try:
        mask = detector(df)
        df.loc[mask, 'rejection_reasons'] += f'{rule_name}|'
        df.loc[mask, 'security_risk_score'] += severity_score(severity)
        df.loc[mask & (df['severity'] != 'Critical'), 'severity'] = severity
        rejection_log[rule_name] += mask.sum()
    except Exception as e:
        pass

approved_df = df[df['rejection_reasons'] == ''].copy()
rejected_df = df[df['rejection_reasons'] != ''].copy()
approved_df['rejection_reasons'] = 'Approved'

print(f"Security rules rejected {len(rejected_df)} flows ({len(rejected_df)/len(df)*100:.2f}%)")
print(f"Remaining flows for ML analysis: {len(approved_df)}")

if approved_df.empty:
    print("All flows rejected by security rules. No data for ML analysis.")
    exit(0)

scaler = QuantileTransformer(output_distribution='uniform')
scaler.fit(df[selected_features])
approved_df[selected_features] = scaler.transform(approved_df[selected_features])

X_test = approved_df[selected_features].values
predictions = model.predict(X_test)
approved_df['predicted_label'] = label_encoder.inverse_transform(predictions)
approved_df['prediction_correct'] = approved_df['label'] == approved_df['predicted_label']

uncertain_df = approved_df[~approved_df['prediction_correct']].copy()
if uncertain_df.empty:
    uncertain_df['reconstruction_error'] = pd.Series(dtype=float)
    uncertain_df['stage2_anomaly'] = pd.Series(dtype=bool)
else:
    uncertain_scaled = ae_scaler.transform(uncertain_df[selected_features])
    uncertain_tensor = torch.tensor(uncertain_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = autoencoder(uncertain_tensor)
        reconstruction_errors = ((reconstructed - uncertain_tensor) ** 2).mean(dim=1).cpu().numpy()
    uncertain_df['reconstruction_error'] = reconstruction_errors
    threshold = 0.05
    uncertain_df['stage2_anomaly'] = uncertain_df['reconstruction_error'] > threshold

if not uncertain_df.empty:
    approved_df.loc[uncertain_df.index, 'stage2_anomaly'] = uncertain_df['stage2_anomaly']
approved_df['stage2_anomaly'] = approved_df['stage2_anomaly'].fillna(False)

accuracy = accuracy_score(approved_df['label'], approved_df['predicted_label'])

report = generate_detailed_security_report(
    df=df,
    approved_df=approved_df,
    initial_count=initial_count,
    rejection_log=rejection_log,
    security_rules=SECURITY_RULES,
    model_accuracy=accuracy,
    uncertain_df=uncertain_df
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"security_report_{timestamp}.txt"

with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report)

df.to_csv(f'security_processed_results_{timestamp}.csv', index=False)
print(f"Report generated: {report_filename}")
