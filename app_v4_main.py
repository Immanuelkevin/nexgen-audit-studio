import os
import shutil
import time
import uvicorn
import json
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torchaudio

# Bypass for SpeechBrain error on newer torchaudio versions
if getattr(torchaudio, "list_audio_backends", None) is None:
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from groq import Groq
from fpdf import FPDF
from speechbrain.inference.classifiers import EncoderClassifier

# Disable symlinks for Windows compatibility
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Configuration ---
# Securely load API Key from environment variables for deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[CRITICAL] GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# MongoDB Atlas Configuration for V4
# password has been changed to Imman123
MONGO_URI = "mongodb+srv://immanuelkevin55_db_user:Imman123@cluster0.ylcv4kw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "nexgen_audit_db"
COLLECTION_NAME = "audit_history_v4"

import motor.motor_asyncio
try:
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("[OK] MongoDB Connection Initialized!")
except Exception as e:
    print(f"[ERROR] MongoDB Connection Failed: {e}")

# Local JSON Database Configuration for V4 (Fallback/Local Mirror)
HISTORY_FILE = "v4_history.json"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# Initialize SpeechBrain Emotion Classifier
try:
    print("Loading SpeechBrain Emotion Model ...")
    emotion_classifier = EncoderClassifier.from_hparams(
        source="d:/voice_emotion_project_v3/transcripting_module/pretrained_models/emotion_recognition",
        run_opts={"device": "cpu"}
    )
    print("[OK] SpeechBrain Model Loaded Successfully!")
    SPEECHBRAIN_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] SpeechBrain Model Loading Failed: {str(e)}")
    print("[WARNING] Continuing without voice emotion detection...")
    SPEECHBRAIN_AVAILABLE = False

app = FastAPI()

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NexGen | Contact Center Audit Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            background-color: #0f172a;
            background-image:
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%),
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%),
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
            color: #e2e8f0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .glass-panel {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
        }
        .drop-zone {
            border: 2px dashed rgba(255,255,255,0.12);
            border-radius: 16px;
            transition: all 0.3s;
        }
        .drop-zone.dragover {
            border-color: #10b981;
            background: rgba(16,185,129,0.08);
        }
        .record-pulse {
            animation: recordPulse 1.2s ease-in-out infinite;
        }
        @keyframes recordPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
            50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
        }
        .tab-btn { transition: all 0.25s; }
        .tab-btn.active {
            background: rgba(16,185,129,0.15);
            border-color: rgba(16,185,129,0.5);
            color: #6ee7b7;
        }
    </style>
</head>
<body class="p-4 md:p-8">

<div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-xl shadow-lg shadow-indigo-500/20">📡</div>
            <h1 class="text-xl md:text-2xl font-bold">NexGen <span class="font-light text-slate-400">Customer Care</span> Audit Studio <span class="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded ml-2">v4.1 PRO</span></h1>
        </div>
        <div class="flex items-center gap-4">
            <button onclick="fetchHistory()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold transition-all shadow-lg flex items-center gap-2">
                📊 View History
            </button>
            <button onclick="syncToCloud()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold transition-all shadow-lg flex items-center gap-2">
                ☁️ Sync Cloud
            </button>
            <div class="px-3 py-1 bg-white/10 rounded-full text-[10px] md:text-xs flex items-center gap-2">
                <div class="w-2 h-2 bg-indigo-400 rounded-full animate-pulse"></div>
                NexGen AI Cloud Intelligence
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-12 gap-8">

        <!-- LEFT COLUMN -->
        <div class="col-span-1 md:col-span-3 flex flex-col gap-6">

            <!-- Input Panel -->
            <div class="glass-panel p-6">
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">Audio Input</h3>

                <!-- Tabs -->
                <div class="flex gap-2 mb-5">
                    <button id="tabUpload" onclick="switchTab('upload')"
                        class="tab-btn active flex-1 text-[10px] uppercase font-bold tracking-widest py-2 rounded-xl border border-white/10">
                        📁 Upload
                    </button>
                    <button id="tabRecord" onclick="switchTab('record')"
                        class="tab-btn flex-1 text-[10px] uppercase font-bold tracking-widest py-2 rounded-xl border border-white/10 text-slate-400">
                        🎙️ Record
                    </button>
                </div>

                <!-- Upload Tab -->
                <div id="panelUpload">
                    <input type="file" id="fileInput" class="hidden" accept="audio/*,video/*">
                    <div id="dropZone" class="drop-zone p-5 text-center cursor-pointer" onclick="document.getElementById('fileInput').click()">
                        <div class="text-3xl mb-2">📤</div>
                        <p class="text-xs text-slate-400">Click or drag & drop audio file</p>
                        <p class="text-[10px] text-slate-600 mt-1">MP3, WAV, M4A, WEBM, OGG, MP4</p>
                    </div>
                </div>

                <!-- Record Tab -->
                <div id="panelRecord" class="hidden">
                    <div class="text-center">
                        <!-- Record Button -->
                        <button id="recordBtn" onclick="toggleRecording()"
                            class="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-rose-500 to-red-600 flex items-center justify-center text-3xl shadow-lg hover:scale-105 transition-all duration-300 mb-4">
                            🎙️
                        </button>
                        <!-- Timer -->
                        <div id="recordTimerWrap" class="hidden flex items-center justify-center gap-2 mb-2">
                            <div class="w-2.5 h-2.5 bg-red-500 rounded-full record-pulse"></div>
                            <span id="recordTimer" class="font-mono text-red-400 font-bold text-sm">00:00</span>
                        </div>
                        <p id="recordHint" class="text-[10px] text-slate-500">Click mic to start recording</p>
                    </div>
                </div>

                <!-- Status Bar -->
                <div id="fileStatus" class="mt-4 text-[10px] text-emerald-400 text-center truncate min-h-[14px]"></div>

                <!-- Language Selection -->
                <div class="mb-4">
                    <label class="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-2 block">Detection Mode</label>
                    <select id="langSelect" class="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2 text-xs focus:border-emerald-500/50 outline-none transition-all cursor-pointer">
                        <option value="auto">✨ UNIVERSAL (AUTO → ENGLISH)</option>
                        <option value="kn">KANNADA</option>
                        <option value="ta">TAMIL</option>
                        <option value="te">TELUGU</option>
                        <option value="hi">HINDI</option>
                        <option value="ml">MALAYALAM</option>
                        <option value="en">ENGLISH</option>
                    </select>
                </div>

                <!-- Divider -->
                <div class="my-4 border-t border-white/5"></div>

                <!-- Analyze Button -->
                <button id="startBtn" onclick="startProcessing()"
                    class="w-full bg-gradient-to-r from-emerald-600 to-teal-600 py-3 rounded-xl font-bold shadow-lg shadow-emerald-500/20 transition-all active:scale-95 text-sm">
                    ⚡ START ANALYSIS
                </button>
            </div>

            <!-- Quality Score -->
            <div class="glass-panel p-6 bg-emerald-900/10 border-emerald-500/30">
                <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-4">Agent Quality Score</h3>
                <div class="flex items-end gap-2 mb-6">
                    <span class="text-5xl font-bold text-white" id="qualityScore">--</span>
                    <span class="text-xl text-slate-500 mb-1">/ 10</span>
                </div>
                <div id="criteriaList" class="space-y-5">
                    <div class="text-slate-500 italic text-xs">Awaiting data...</div>
                </div>
            </div>

            <!-- Accuracy -->
            <div class="glass-panel p-6 bg-blue-900/10 border-blue-500/30">
                <h3 class="text-xs font-bold uppercase tracking-wider text-blue-400 mb-4">Transcription Accuracy</h3>
                <div class="flex items-end gap-2 mb-4">
                    <span class="text-4xl font-bold text-white" id="accuracyScore">--</span>
                    <span class="text-lg text-slate-500 mb-1">%</span>
                </div>
                <div class="text-[10px] text-slate-500 uppercase tracking-widest">AI Confidence Level</div>
            </div>

            <!-- Sentiment -->
            <div class="glass-panel p-6 bg-purple-900/10 border-purple-500/30">
                <h3 class="text-xs font-bold uppercase tracking-wider text-purple-400 mb-4">Sentiment Analysis</h3>
                <div class="space-y-4">
                    <div class="flex justify-between items-end">
                        <span class="text-[10px] text-slate-400 uppercase">Interaction Tone</span>
                        <span class="text-xs uppercase font-bold text-purple-300" id="sentimentLabel">--</span>
                    </div>
                    <div class="flex gap-1 h-2 rounded-full overflow-hidden bg-slate-800">
                        <div id="sentNeg" class="bg-red-500 w-0 transition-all duration-700 h-full"></div>
                        <div id="sentNeu" class="bg-slate-500 w-0 transition-all duration-700 h-full"></div>
                        <div id="sentPos" class="bg-emerald-500 w-0 transition-all duration-700 h-full"></div>
                    </div>
                </div>
            </div>

            <!-- Risk & Compliance -->
            <div class="glass-panel p-6 bg-rose-900/10 border-rose-500/30">
                <h3 class="text-xs font-bold uppercase tracking-wider text-rose-400 mb-4 flex items-center gap-2">
                    🛡️ Risk & Compliance
                </h3>
                <div id="riskFlags" class="space-y-2">
                    <div class="text-[10px] text-slate-600 italic">No risks detected...</div>
                </div>
            </div>
        </div>

        <!-- RIGHT COLUMN -->
        <div class="col-span-1 md:col-span-9 flex flex-col gap-6 min-h-[500px]">

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Tone Chart -->
                <div class="glass-panel p-6 bg-indigo-900/10 border-indigo-500/30">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
                            Voice Tone Analysis
                        </h3>
                        <span id="dominantEmotion" class="text-[10px] font-bold uppercase tracking-widest text-indigo-300 bg-indigo-500/10 px-3 py-1 rounded-full">Awaiting...</span>
                    </div>
                    <div class="relative" style="height:180px;">
                        <canvas id="toneChart"></canvas>
                    </div>
                </div>

                <!-- Sentiment Journey -->
                <div class="glass-panel p-6 bg-emerald-900/10 border-emerald-500/30">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-4 flex items-center gap-2">
                        📈 Sentiment Journey Arc
                    </h3>
                    <div class="relative" style="height:180px;">
                        <canvas id="journeyChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Interaction Analysis -->
            <div class="glass-panel p-6 flex flex-col relative overflow-hidden" style="min-height:380px;">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                    <h2 class="text-lg font-semibold">Interaction Analysis</h2>
                    <button id="downloadPdfBtn" onclick="downloadReport()" class="hidden flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-2 rounded-xl text-xs font-bold transition-all border border-white/5 w-full sm:w-auto justify-center">
                        📥 DOWNLOAD AI AUDIT PDF
                    </button>
                </div>
                
                <!-- Waveform Container -->
                <div id="waveformControl" class="hidden mb-6 p-4 bg-black/40 rounded-2xl border border-white/5">
                    <div id="waveform" class="w-full"></div>
                    <div class="flex justify-center gap-4 mt-3">
                        <button onclick="wavesurfer.playPause()" class="text-emerald-400 hover:text-emerald-300 transition-colors text-xs font-bold uppercase tracking-widest">
                            <span id="playIcon">▶️ Play / Pause</span>
                        </button>
                    </div>
                </div>
                
                <!-- Loader Overlay (Moved Up) -->
                <div id="loadingIndicator" class="hidden absolute inset-0 flex flex-col items-center justify-center bg-slate-900/90 backdrop-blur-xl z-50">
                    <div class="relative w-24 h-24 mb-10">
                        <div class="absolute inset-0 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin"></div>
                        <div class="absolute inset-3 border-4 border-teal-500/10 border-b-teal-500 rounded-full animate-spin [animation-duration:1.5s]"></div>
                    </div>
                    <div class="w-64 h-2 bg-white/5 rounded-full mb-8 overflow-hidden border border-white/10">
                        <div id="realProgressBar" class="h-full bg-emerald-500 w-0 transition-all duration-500"></div>
                    </div>
                    <div class="space-y-4 w-56 text-[10px] tracking-widest uppercase text-center font-bold">
                        <div id="step1" class="text-slate-500">Groq Cloud Upload</div>
                        <div id="step2" class="text-slate-500">Neural Decoding</div>
                        <div id="step3" class="text-slate-500">AI Quality Audit</div>
                    </div>
                </div>

                <div id="transcriptContainer" class="flex-1 overflow-y-auto text-sm leading-relaxed p-6 bg-black/20 rounded-2xl font-light whitespace-pre-wrap text-slate-300 relative" style="min-height:280px;">
                    <div id="placeholderApp" class="flex items-center justify-center h-full text-slate-600 italic">
                        Upload or record audio, then hit Start Analysis...
                    </div>
                    <div id="resultsContent" class="hidden h-full"></div>
                    <audio id="notifSound" src="https://cdn.freesound.org/previews/263/263125_4818617-lq.mp3" preload="auto"></audio>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- History Modal -->
<div id="historyModal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm p-4">
    <div class="glass-panel w-full max-w-4xl max-h-[80vh] flex flex-col bg-slate-800 border-slate-600">
        <div class="flex justify-between items-center p-6 border-b border-white/10">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">📊 Audit History Database</h2>
            <button onclick="document.getElementById('historyModal').classList.add('hidden')" class="text-slate-400 hover:text-white text-2xl font-bold">&times;</button>
        </div>
        <div class="p-6 overflow-y-auto flex-1">
            <table class="w-full text-left text-sm text-slate-300">
                <thead class="text-xs uppercase bg-white/5 text-slate-400">
                    <tr>
                        <th class="px-4 py-3 rounded-tl-lg">Date/Time</th>
                        <th class="px-4 py-3">Language</th>
                        <th class="px-4 py-3 text-center">Score</th>
                        <th class="px-4 py-3">Risk</th>
                        <th class="px-4 py-3 rounded-tr-lg">Sentiment</th>
                    </tr>
                </thead>
                <tbody id="historyTableBody">
                    <tr><td colspan="5" class="px-4 py-6 text-center italic text-slate-500">Loading history...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
    async function fetchHistory() {
        const modal = document.getElementById('historyModal');
        const tbody = document.getElementById('historyTableBody');
        modal.classList.remove('hidden');
        tbody.innerHTML = '<tr><td colspan="5" class="px-4 py-6 text-center italic text-slate-500">Loading history...</td></tr>';
        
        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            
            if (!resp.ok || data.error) {
                throw new Error(data.error || `HTTP ${resp.status}`);
            }

            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="px-4 py-6 text-center italic text-slate-500">No records found.</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.map(item => {
                const date = new Date(item.timestamp * 1000).toLocaleString();
                const score = item.audit ? item.audit.total_score : '--';
                const risks = item.audit && item.audit.risk_flags && item.audit.risk_flags.length > 0 
                                ? `<span class="text-rose-400 font-bold">${item.audit.risk_flags.length} Flags</span>` 
                                : `<span class="text-emerald-400">Safe</span>`;
                const sent = item.audit && item.audit.sentiment ? item.audit.sentiment.label.toUpperCase() : '--';
                
                return `<tr class="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td class="px-4 py-3">${date}</td>
                    <td class="px-4 py-3">${item.detected_language || 'UNKNOWN'}</td>
                    <td class="px-4 py-3 font-bold text-center text-lg text-emerald-300">${score}<span class="text-xs text-slate-500">/10</span></td>
                    <td class="px-4 py-3">${risks}</td>
                    <td class="px-4 py-3 text-xs tracking-wider">${sent}</td>
                </tr>`;
            }).join('');
        } catch (e) {
            tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-6 text-center text-red-500">Error: ${e.message}</td></tr>`;
        }
    }

    async function syncToCloud() {
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = '⌛ Syncing...';
        btn.disabled = true;
        try {
            const resp = await fetch('/api/sync_to_mongo');
            const data = await resp.json();
            if (resp.ok) {
                alert(data.message);
            } else {
                alert('Sync Error: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            alert('Network Error: ' + e.message);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    // ── Chart Setup ──────────────────────────────────────────────────────────
    const EMOTION_LABELS = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised'];
    const EMOTION_COLORS_BG = [
        'rgba(239,68,68,0.7)','rgba(99,102,241,0.7)','rgba(168,85,247,0.7)','rgba(251,146,60,0.7)',
        'rgba(34,197,94,0.7)','rgba(100,116,139,0.7)','rgba(59,130,246,0.7)','rgba(251,191,36,0.7)'
    ];
    const EMOTION_COLORS_BORDER = [
        'rgba(239,68,68,1)','rgba(99,102,241,1)','rgba(168,85,247,1)','rgba(251,146,60,1)',
        'rgba(34,197,94,1)','rgba(100,116,139,1)','rgba(59,130,246,1)','rgba(251,191,36,1)'
    ];

    const toneCtx = document.getElementById('toneChart').getContext('2d');
    const toneChart = new Chart(toneCtx, {
        type: 'bar',
        data: {
            labels: EMOTION_LABELS,
            datasets: [{
                label: 'Score',
                data: [0,0,0,0,0,0,0,0],
                backgroundColor: EMOTION_COLORS_BG,
                borderColor: EMOTION_COLORS_BORDER,
                borderWidth: 2, borderRadius: 6, borderSkipped: false
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 600 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: ctx => ` Score: ${ctx.parsed.y.toFixed(3)}` },
                    backgroundColor: 'rgba(15,23,42,0.95)', borderColor: 'rgba(99,102,241,0.5)',
                    borderWidth: 1, titleColor: '#a5b4fc', bodyColor: '#e2e8f0'
                }
            },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10, family: 'Outfit' } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { beginAtZero: true, max: 0.3, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    const journeyCtx = document.getElementById('journeyChart').getContext('2d');
    const journeyChart = new Chart(journeyCtx, {
        type: 'line',
        data: {
            labels: ['Start', 'S1', 'S2', 'S3', 'S4', 'S5', 'End'],
            datasets: [{
                label: 'Sentiment',
                data: [0, 0, 0, 0, 0, 0, 0],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16,185,129,0.1)',
                borderWidth: 3, fill: true, tension: 0.4, pointRadius: 4, pointBackgroundColor: '#10b981'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                y: { min: -1.1, max: 1.1, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    function updateToneChart(emotionScores) {
        if (!emotionScores) return;
        const newData = EMOTION_LABELS.map(l => emotionScores[l] || 0);
        toneChart.data.datasets[0].data = newData;
        toneChart.options.scales.y.max = Math.max(Math.max(...newData) * 1.3, 0.15);
        toneChart.update();
        const maxVal = Math.max(...newData);
        const dominantIdx = newData.indexOf(maxVal);
        const badge = document.getElementById('dominantEmotion');
        badge.innerText = EMOTION_LABELS[dominantIdx].toUpperCase() + ' · ' + maxVal.toFixed(3);
        badge.style.color = EMOTION_COLORS_BORDER[dominantIdx];
    }

    function updateJourneyChart(journeyData) {
        if (!journeyData || !Array.isArray(journeyData)) return;
        journeyChart.data.labels = journeyData.map((_, i) => i === 0 ? 'Start' : (i === journeyData.length - 1 ? 'End' : 'Seg ' + i));
        journeyChart.data.datasets[0].data = journeyData;
        journeyChart.update();
    }

    // ── Tab Switcher ─────────────────────────────────────────────────────────
    function switchTab(tab) {
        document.getElementById('panelUpload').classList.toggle('hidden', tab !== 'upload');
        document.getElementById('panelRecord').classList.toggle('hidden', tab !== 'record');
        document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
        document.getElementById('tabRecord').classList.toggle('active', tab === 'record');
        if (tab === 'upload') {
            document.getElementById('tabUpload').classList.add('text-emerald-300');
            document.getElementById('tabRecord').classList.remove('text-emerald-300');
        } else {
            document.getElementById('tabRecord').classList.add('text-emerald-300');
            document.getElementById('tabUpload').classList.remove('text-emerald-300');
        }
    }

    // ── File Upload ───────────────────────────────────────────────────────────
    let selectedFile = null;

    document.getElementById('fileInput').onchange = (e) => {
        selectedFile = e.target.files[0];
        if (selectedFile) setFileStatus('📁 ' + selectedFile.name);
    };

    // Drag & Drop
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) { selectedFile = file; setFileStatus('📁 ' + file.name); }
    });

    function setFileStatus(msg) {
        document.getElementById('fileStatus').innerText = msg;
    }

    // ── Recording ─────────────────────────────────────────────────────────────
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let recordingStream;
    let recordingTimerInterval;
    let recordingSeconds = 0;

    async function toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    }

    async function startRecording() {
        try {
            recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            try {
                mediaRecorder = new MediaRecorder(recordingStream, { mimeType: 'audio/webm' });
            } catch(e) {
                mediaRecorder = new MediaRecorder(recordingStream);
            }
            audioChunks = [];

            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                selectedFile = new File([blob], 'recording.webm', { type: 'audio/webm' });
                setFileStatus('🎙️ Recording ready · ' + formatTime(recordingSeconds));
                recordingStream.getTracks().forEach(t => t.stop());
            };

            mediaRecorder.start(100);
            isRecording = true;
            recordingSeconds = 0;

            // UI: recording state
            const btn = document.getElementById('recordBtn');
            btn.innerHTML = '⏹️';
            btn.className = 'w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-3xl shadow-lg hover:scale-105 transition-all duration-300 mb-4 record-pulse';
            document.getElementById('recordTimerWrap').classList.remove('hidden');
            document.getElementById('recordHint').innerText = 'Recording... click to stop';
            document.getElementById('fileStatus').innerText = '';

            recordingTimerInterval = setInterval(() => {
                recordingSeconds++;
                document.getElementById('recordTimer').innerText = formatTime(recordingSeconds);
            }, 1000);

        } catch(err) {
            alert('Microphone access denied. Please allow microphone permission in browser settings.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        isRecording = false;
        clearInterval(recordingTimerInterval);

        // UI: idle state
        const btn = document.getElementById('recordBtn');
        btn.innerHTML = '🎙️';
        btn.className = 'w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-rose-500 to-red-600 flex items-center justify-center text-3xl shadow-lg hover:scale-105 transition-all duration-300 mb-4';
        document.getElementById('recordTimerWrap').classList.add('hidden');
        document.getElementById('recordHint').innerText = 'Click mic to start recording';
    }

    function formatTime(secs) {
        const m = Math.floor(secs / 60).toString().padStart(2, '0');
        const s = (secs % 60).toString().padStart(2, '0');
        return m + ':' + s;
    }

    // ── Global State ──────────────────────────────────────────────────────────
    let lastAnalysisResult = null;
    let wavesurfer = null;

    function initWaveform(audioUrl) {
        if (wavesurfer) wavesurfer.destroy();
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgba(16, 185, 129, 0.3)',
            progressColor: 'rgba(5, 150, 105, 1)',
            cursorColor: '#6ee7b7',
            barWidth: 2,
            barGap: 3,
            barRadius: 4,
            responsive: true,
            height: 80,
            cursorWidth: 2
        });
        wavesurfer.load(audioUrl);
        document.getElementById('waveformControl').classList.remove('hidden');
        
        wavesurfer.on('play', () => document.getElementById('playIcon').innerText = '⏸️ Pause');
        wavesurfer.on('pause', () => document.getElementById('playIcon').innerText = '▶️ Play');
    }

    async function downloadReport() {
        if (!lastAnalysisResult) return;
        const btn = document.getElementById('downloadPdfBtn');
        const originalText = btn.innerText;
        btn.innerText = '⌛ GENERATING...';
        btn.disabled = true;

        try {
            const resp = await fetch('/generate_pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastAnalysisResult)
            });
            if (!resp.ok) throw new Error("PDF Generation failed");
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `NexGen_Audit_${new Date().getTime()}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch(e) {
            alert(e.message);
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    }

    // ── Analysis ──────────────────────────────────────────────────────────────
    async function startProcessing() {
        if (!selectedFile) {
            alert('Please upload or record audio first.');
            return;
        }
        if (isRecording) {
            alert('Please stop recording first before analysing.');
            return;
        }

        const btn = document.getElementById('startBtn');
        const placeholder = document.getElementById('placeholderApp');
        const loader = document.getElementById('loadingIndicator');
        const results = document.getElementById('resultsContent');
        const progBar = document.getElementById('realProgressBar');

        // Load Waveform local url
        const audioUrl = URL.createObjectURL(selectedFile);
        initWaveform(audioUrl);

        btn.disabled = true;
        placeholder.classList.add('hidden');
        loader.classList.remove('hidden');
        results.classList.add('hidden');
        document.getElementById('downloadPdfBtn').classList.add('hidden');
        results.innerHTML = '';
        progBar.style.width = '5%';
        updateSteps(1);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('lang', document.getElementById('langSelect').value);

        try {
            let prog = 5;
            const progInterval = setInterval(() => {
                if (prog < 90) {
                    prog += 2;
                    progBar.style.width = prog + '%';
                    if (prog > 45) updateSteps(2);
                    if (prog > 80) updateSteps(3);
                }
            }, 400);

            const response = await fetch('/transcribe', { method: 'POST', body: formData });
            const data = await response.json();
            clearInterval(progInterval);
            if (!response.ok) throw new Error(data.error);

            lastAnalysisResult = data; // Store for PDF
            progBar.style.width = '100%';
            updateSteps(3);

            setTimeout(() => {
                loader.classList.add('hidden');
                results.classList.remove('hidden');
                document.getElementById('downloadPdfBtn').classList.remove('hidden');
                document.getElementById('notifSound').play().catch(() => {});

                results.innerHTML = `<div class="mb-10">
                    <h3 class="text-emerald-400 font-bold mb-4 uppercase text-[10px] tracking-[0.2em]">Transcript:</h3>
                    <p class="text-slate-200 leading-relaxed font-light text-lg tracking-wide">${data.text}</p>
                </div>`;

                if (data.audit) {
                    const langDisplay = data.detected_language || (data.language_used || 'AUTO');
                    results.innerHTML += `<div class="p-8 bg-white/5 rounded-[32px] border border-white/10 mt-6 shadow-2xl">
                        <h3 class="text-emerald-400 font-bold mb-4 flex items-center gap-3">
                            <span class="bg-emerald-500/20 p-2 rounded-xl">🛡️</span> AI Auditor Insight
                        </h3>
                        <p class="text-slate-300 text-lg italic leading-relaxed font-light mb-6">"${data.audit.summary}"</p>
                        <div class="pt-6 border-t border-white/5 flex justify-between items-center text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                            <span>Language: ${langDisplay}</span>
                            <span>Engine: Llama-3.3-70B</span>
                        </div>
                    </div>
                    
                    <div class="p-8 bg-black/40 rounded-[32px] border border-blue-500/20 mt-6 shadow-[0_0_20px_rgba(59,130,246,0.15)] relative overflow-hidden">
                        <div class="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-transparent"></div>
                        <h3 class="text-blue-400 font-bold mb-4 flex items-center gap-3 relative z-10 text-[10px] uppercase tracking-[0.2em]">
                            <span class="text-xl animate-pulse">🎙️</span> SpeechBrain Acoustic Analysis
                        </h3>
                        <div class="flex justify-between items-center relative z-10">
                            <span class="text-slate-300 font-light tracking-wider uppercase text-sm">Detected Voice Tone:</span>
                            <span class="text-3xl font-black ${data.voice_emotion && (data.voice_emotion.toLowerCase() === 'angry' || data.voice_emotion.toLowerCase() === 'sad') ? 'text-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'text-blue-400 drop-shadow-[0_0_8px_rgba(96,165,250,0.8)]' }">${data.voice_emotion || 'Unknown'}</span>
                        </div>
                        <div class="mt-4 pt-4 border-t border-white/5 relative z-10">
                            <div class="flex justify-between text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2">
                                <span>Acoustic Confidence</span>
                                <span>${data.voice_confidence ? Math.round(data.voice_confidence) : 0}%</span>
                            </div>
                            <div class="h-1 bg-white/5 rounded-full overflow-hidden">
                                <div class="h-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)] transition-all" style="width:${data.voice_confidence || 0}%"></div>
                            </div>
                        </div>
                    </div>`;

                    document.getElementById('qualityScore').innerText = data.audit.total_score;

                    const criteriaHtml = Object.entries(data.audit.criteria_breakdown).map(([key, val]) => `
                        <div>
                            <div class="flex justify-between items-center mb-1">
                                <span class="capitalize text-slate-400 text-[10px] uppercase font-black tracking-widest">${key.replace(/_/g,' ')}</span>
                                <span class="${val > 1 ? 'text-emerald-400' : 'text-red-400'} font-bold">${val}/2</span>
                            </div>
                            <div class="h-1 bg-white/5 rounded-full overflow-hidden">
                                <div class="h-full ${val > 1 ? 'bg-emerald-500' : 'bg-red-500'} transition-all duration-1000 shadow-[0_0_10px_rgba(16,185,129,0.3)]" style="width:${val*50}%"></div>
                            </div>
                        </div>`).join('');
                    document.getElementById('criteriaList').innerHTML = `<div class="space-y-6">${criteriaHtml}</div>`;

                    const s = data.audit.sentiment;
                    document.getElementById('sentimentLabel').innerText = s.label;
                    document.getElementById('sentNeg').style.width = (s.score_neg * 100) + '%';
                    document.getElementById('sentNeu').style.width = (s.score_neu * 100) + '%';
                    document.getElementById('sentPos').style.width = (s.score_pos * 100) + '%';

                    document.getElementById('accuracyScore').innerText = data.audit.transcription_confidence || '--';

                    if (data.audit.emotion_scores) updateToneChart(data.audit.emotion_scores);
                    if (data.audit.sentiment_journey) updateJourneyChart(data.audit.sentiment_journey);

                    // Risk Flags
                    const riskContainer = document.getElementById('riskFlags');
                    const flags = data.audit.risk_flags || [];
                    if (flags.length > 0) {
                        riskContainer.innerHTML = flags.map(f => `
                            <div class="flex items-center gap-2 bg-rose-500/10 border border-rose-500/20 p-2 rounded-lg text-[10px] text-rose-300 font-bold uppercase tracking-wider">
                                <span class="animate-pulse text-rose-500">🚩</span> ${f}
                            </div>
                        `).join('');
                    } else {
                        riskContainer.innerHTML = '<div class="text-[10px] text-emerald-400 font-bold uppercase tracking-widest flex items-center gap-2">✅ ALL COMPLIANT</div>';
                    }
                }
            }, 500);

        } catch (error) {
            alert('Analysis Error: ' + error.message);
            loader.classList.add('hidden');
            placeholder.classList.remove('hidden');
        } finally {
            btn.disabled = false;
        }
    }

    function updateSteps(step) {
        for (let i = 1; i <= 3; i++) {
            const el = document.getElementById('step' + i);
            if (!el) continue;
            if (i < step) el.className = 'text-emerald-400 font-bold';
            else if (i === step) el.className = 'text-amber-400 font-bold animate-pulse';
            else el.className = 'text-slate-500';
        }
    }
</script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form("auto")):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Groq Transcription with Auto Language Detection
        start_time = time.time()
        with open(temp_filename, "rb") as audio_file:
            # Build multilingual prompt covering all 5 target languages.
            # NOTE: Ending the prompt with natural call-centre closing words primes
            # Whisper to expect those phrases instead of YouTube-style endings.
            multilingual_prompt = (
                "Contact Center customer service call - Agent side only. "
                "The speaker is a customer care agent using Tamil, Telugu, Hindi, Malayalam, Kannada, or English. "
                "Common words: hello, hi, yes, no, ok, okay, sure, thanks, thank you, "
                "The agent is assisting a customer with professional services."
            )

            trans_params = {
                "model": "whisper-large-v3",
                "file": audio_file,
                "prompt": multilingual_prompt,
                "temperature": 0,
                "response_format": "verbose_json",
            }

            # AUTO Mode: Perform translation to English (Universal Mode)
            # Specific Language Mode: Perform transcription in native script
            if lang == "auto":
                transcript_response = client.audio.translations.create(**trans_params)
            else:
                trans_params["language"] = lang
                transcript_response = client.audio.transcriptions.create(**trans_params)

        transcribed_text = transcript_response.text
        # Get the language Whisper auto-detected
        detected_lang_code = getattr(transcript_response, "language", lang)

        # --- Hallucination Filter ---
        # Whisper sometimes outputs YouTube/media-style phrases for short audio.
        # Detect and clean these known patterns.
        HALLUCINATION_PATTERNS = [
            "thank you for watching",
            "thanks for watching",
            "please subscribe",
            "like and subscribe",
            "don't forget to subscribe",
            "see you in the next video",
            "see you next time",
            "subtitles by",
            "transcribed by",
            "service is not available. please wait for a minute."
        ]
        lower_text = transcribed_text.strip().lower()
        for pattern in HALLUCINATION_PATTERNS:
            if lower_text == pattern or lower_text == pattern + ".":
                print(f"-> Hallucination detected: '{transcribed_text}' -> replacing with 'Thanks.'")
                transcribed_text = "Thanks."
                break

        print(f"-> Detected language: {detected_lang_code} | Text: {transcribed_text[:80]}")

        # 1.5 SpeechBrain Audio Emotion Detection
        if SPEECHBRAIN_AVAILABLE:
            print("-> Running SpeechBrain Emotion Analysis...")
            try:
                import soundfile as sf
                import numpy as np
                import subprocess
                
                # Convert the incoming audio (usually WebM from browser) to 16kHz WAV format required by SpeechBrain
                wav_filename = temp_filename + ".wav"
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_filename, 
                    "-ar", "16000", "-ac", "1", wav_filename
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                # Run SpeechBrain native classifier directly on the WAV file
                out_prob, score, index, text_lab = emotion_classifier.classify_file(wav_filename)
                
                # Extract predictions
                detected_emotion = text_lab[0]
                confidence = score[0].item()
                
                
                # Map to readable labels
                emotion_map = {
                    'ang': 'Angry',
                    'hap': 'Happy',
                    'sad': 'Sad',
                    'neu': 'Neutral',
                    'fea': 'Fearful'
                }
                voice_emotion = emotion_map.get(detected_emotion, detected_emotion.capitalize())
                voice_confidence = confidence * 100
                
                print(f"-> Voice Emotion: {voice_emotion} ({voice_confidence:.1f}% confidence)")
            except Exception as e:
                print(f"-> SpeechBrain Error: {str(e)}")
                import traceback
                traceback.print_exc()
                voice_emotion = "Unknown"
                voice_confidence = 0
        else:
            voice_emotion = "Not Available"
            voice_confidence = 0
            print("-> SpeechBrain not available, skipping voice emotion detection")

        # 2. Audit Logic
        system_prompt = """
        You are an Expert Quality & Compliance Auditor for NexGen Customer Care.
        Your goal is to evaluate the performance of the Customer Care Service Agent ONLY based on their spoken side of the call.
        
        STRICT RULES:
        1. YOU ARE REVIEWING THE AGENT'S AUDIO ONLY. 
        2. DO NOT comment on the customer. Do not guess what the customer said.
        3. Do not assume the customer was "abusive" or "angry" unless the agent explicitly says "Sir, please do not use such language".
        4. Focus 100% on the Agent's adherence to professional scripts and brand guidelines.

        CRITICAL SECURITY STEP: Redact all PII (Mobile numbers, Emails, Aadhaar/IDs, Credit Cards, specific addresses) spoken by the agent. Replace them with [REDACTED].

        CRITICAL COMPLIANCE STEP: Identify "Agent Risk Flags". 
        A Risk Flag is raised if:
        - Agent mentions competitors or local workshops.
        - Agent is unprofessional, rude, or loses patience.
        - Agent fails to mention key "NexGen" brand names.
        - Agent provides incorrect info or seems uncertain (e.g., "I don't know," "Maybe").

        Evaluation Grid (10 pts Total):
        1. Brand Greeting: Specifically mentioned "NexGen Solutions" professionally (2 pts)
        2. Clarity of Solution: Agent provided clear, logical information/solution (2 pts)
        3. Professional Tone: Empathy, polite language, and patient tone (2 pts)
        4. Compliance: No mention of competitors or unprofessional remarks (2 pts)
        5. Quality Closure: Professional sign-off and clear next steps (2 pts)

        Return JSON ONLY:
        {
            "total_score": NUMBER,
            "criteria_breakdown": {"brand_greeting": X, "solution_clarity": X, "professional_tone": X, "compliance": X, "quality_closure": X},
            "summary": "Focus ONLY on the AGENT. 2 Sentences max.",
            "sentiment": {"label": "pos/neu/neg", "score_pos": X, "score_neu": X, "score_neg": X},
            "redacted_transcript": "AGENT TRANSCRIPT WITH [REDACTED] TAGS",
            "risk_flags": ["LIST", "OF", "AGENT", "RED", "FLAGS", "OR", "EMPTY"],
            "sentiment_journey": [6-8 numbers representing the Agent's emotional flow],
            "transcription_confidence": NUMBER (0-100),
            "emotion_scores": {
                "angry": FLOAT, "calm": FLOAT, "disgust": FLOAT, "fearful": FLOAT,
                "happy": FLOAT, "neutral": FLOAT, "sad": FLOAT, "surprised": FLOAT
            }
        }
        """
        
        audit_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript:\n\n{transcribed_text}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        audit_data = json.loads(audit_response.choices[0].message.content)
        
        # Use redacted transcript for the UI if provided
        display_text = audit_data.get("redacted_transcript", transcribed_text)
        
        # Map language code to human-readable name
        LANG_MAP = {
            "ta": "TAMIL", "te": "TELUGU", "hi": "HINDI",
            "ml": "MALAYALAM", "kn": "KANNADA", "en": "ENGLISH", "auto": "UNIVERSAL ENGLISH"
        }
        
        # If auto was used, detected_lang_code might be 'auto' or 'english'
        if lang == "auto":
            lang_display = "UNIVERSAL (ENGLISH)"
        else:
            lang_display = LANG_MAP.get(detected_lang_code, detected_lang_code.upper())

        result_data = {
            "text": display_text,
            "audit": audit_data,
            "voice_emotion": voice_emotion,
            "voice_confidence": round(voice_confidence, 1),
            "detected_language": lang_display,
            "detected_language_code": detected_lang_code,
            "duration": round(time.time() - start_time, 2),
            "timestamp": time.time()  # Useful for database sorting
        }

        # 3. Save to History (Local & MongoDB)
        try:
            db_record = result_data.copy()
            
            # A. Save to Local JSON History V4 File
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            history_data.append(db_record)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            print("-> Successfully saved audit to local V4 history.")
            
            # B. Save to MongoDB Atlas
            await collection.insert_one(db_record)
            print("-> Successfully synced audit to MongoDB Atlas.")

        except Exception as e:
            print(f"-> Failed to save to history: {e}")

        return JSONResponse(result_data)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        if os.path.exists(temp_filename + ".wav"): os.remove(temp_filename + ".wav")

@app.get("/api/history")
async def get_history():
    try:
        # Try fetching from MongoDB first for most recent data across instances
        try:
            cursor = collection.find().sort("timestamp", -1).limit(50)
            audits = await cursor.to_list(length=50)
            if audits:
                # Remove MongoDB _id for JSON serialization
                for a in audits: a.pop("_id", None)
                return JSONResponse(audits)
        except Exception as mongo_err:
            print(f"-> MongoDB Fetch Failed, falling back to local JSON: {mongo_err}")

        # Fallback to local JSON
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            audits = json.load(f)
        audits.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return JSONResponse(audits[:50])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/sync_to_mongo")
async def sync_to_mongo():
    """Manual trigger to sync local JSON history to MongoDB Atlas."""
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            local_history = json.load(f)
        
        if not local_history:
            return JSONResponse({"message": "No local history to sync."})

        # Get existing timestamps in MongoDB to avoid duplicates
        existing_timestamps = await collection.distinct("timestamp")
        
        to_insert = [rec for rec in local_history if rec.get("timestamp") not in existing_timestamps]
        
        if to_insert:
            await collection.insert_many(to_insert)
            return JSONResponse({"message": f"Successfully synced {len(to_insert)} records to MongoDB Atlas."})
        else:
            return JSONResponse({"message": "All local records are already in MongoDB."})
            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/generate_pdf")
async def generate_pdf(data: dict):
    """Generates a professional PDF report from analysis data."""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Header / Branding
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(79, 70, 229) # Indigo 600
        pdf.cell(0, 20, "NexGen Customer Care Audit Report", ln=True, align="C")
        
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(100, 116, 139) # Slate 500
        pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(10)
        
        # Overview Section
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(30, 41, 59) # Slate 800
        pdf.cell(0, 12, "Interaction Summary", ln=True)
        
        pdf.set_font("helvetica", "", 12)
        pdf.set_text_color(51, 65, 85) # Slate 700
        summary = data.get("audit", {}).get("summary", "No summary provided.")
        pdf.multi_cell(0, 8, summary)
        pdf.ln(10)
        
        # Quality Scores
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 12, "Agent Performance", ln=True)
        
        total_score = data.get("audit", {}).get("total_score", "0")
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(0, 10, f"Total Quality Score: {total_score} / 10", ln=True)
        
        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(51, 65, 85)
        breakdown = data.get("audit", {}).get("criteria_breakdown", {})
        for criterion, score in breakdown.items():
            name = criterion.replace("_", " ").capitalize()
            pdf.cell(80, 8, f"- {name}:", border=0)
            pdf.cell(0, 8, f"{score} / 2", border=0, ln=True)
        pdf.ln(10)
        
        # Risk & Compliance
        risk_flags = data.get("audit", {}).get("risk_flags", [])
        if risk_flags:
            pdf.set_font("helvetica", "B", 16)
            pdf.set_text_color(225, 29, 72) # Rose 600
            pdf.cell(0, 12, "Compliance Risk Flags Detected", ln=True)
            pdf.set_font("helvetica", "B", 10)
            for flag in risk_flags:
                pdf.cell(0, 8, f"ALERT: {flag}", ln=True)
            pdf.ln(5)

        # Sentiment & Transcript
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 12, "Transcription & Insights", ln=True)
        
        sentiment = data.get("audit", {}).get("sentiment", {}).get("label", "Unknown").upper()
        confidence = data.get("audit", {}).get("transcription_confidence", "0")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 8, f"Detected Sentiment: {sentiment}", ln=True)
        pdf.cell(0, 8, f"Transcription Confidence: {confidence}%", ln=True)
        pdf.ln(5)
        
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 10, "Redacted Transcript:", ln=True)
        
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        transcript = data.get("text", "")
        # Sanitize transcript for latin-1 (fpdf standard font limitation)
        safe_transcript = transcript.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_transcript)
        
        # Footer
        pdf.set_y(-30)
        pdf.set_font("helvetica", "I", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 10, "Confidential - For Internal Use Only - NexGen AI Solutions", ln=True, align="C")

        pdf_output = "audit_report.pdf"
        pdf.output(pdf_output)
        
        return FileResponse(pdf_output, filename="NexGen_Audit_Report.pdf", media_type="application/pdf")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
