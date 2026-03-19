import { useState } from 'react'
import { useAnalysis } from './hooks/useAnalysis.js'
import UploadPanel from './UploadPanel.jsx'
import Phase1 from './Phase1.jsx'
import Phase2 from './Phase2.jsx'
import Phase3 from './Phase3.jsx'
import s from './App.module.css'

const TABS = [
  { emoji: '🧠', label: 'AI Engine',       cls: 't1' },
  { emoji: '🧬', label: 'Virtual Biopsy',  cls: 't2' },
  { emoji: '👤', label: 'Digital Twin',    cls: 't3' },
]

const FORM_INIT = { age: '', gender: '', skinType: '', location: '', history: '', notes: '' }

export default function App() {
  const [imagePreview, setImagePreview] = useState(null)
  const [imageFile,    setImageFile]    = useState(null)
  const [form,         setForm]         = useState(FORM_INIT)

  const {
    results, loading, activeTab, setActiveTab,
    steps, error, backendOnline, runAnalysis
  } = useAnalysis()

  function handleImage(file) {
    if (!file) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
  }

  function clearImage() {
    setImageFile(null)
    setImagePreview(null)
  }

  function handleFormChange(key, value) {
    setForm(f => ({ ...f, [key]: value }))
  }

  function handleRun() {
    runAnalysis(imageFile, null, form)
  }

  return (
    <div className={s.app}>
      {/* ── Header ── */}
      <header className={s.header}>
        <div className={s.logo}>
          <div className={s.logoMark}>⬡</div>
          <div>
            <div className={s.logoText}>Derm<span>AI</span></div>
            <div className={s.logoSub}>Precision Oncology Platform</div>
          </div>
        </div>
        <div className={s.headerRight}>
          {backendOnline !== null && (
            <div className={`${s.statusPill} ${backendOnline ? s.online : s.offline}`}>
              <span className={s.statusDot} />
              {backendOnline ? 'ML Model Online' : 'Demo Mode'}
            </div>
          )}
          <div className={s.badges}>
            <span className={`${s.badge} ${s.b1}`}>01 Engine</span>
            <span className={`${s.badge} ${s.b2}`}>02 Biopsy</span>
            <span className={`${s.badge} ${s.b3}`}>03 Twin</span>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <div className={s.hero}>
        <div className={s.eyebrow}>Non-Invasive · Molecular · Personalized</div>
        <h1 className={s.heroTitle}>
          From Photo to <em>Precision Treatment</em> in Seconds
        </h1>
        <p className={s.heroSub}>
          Upload a skin lesion image + patient metadata. 3-phase AI platform
          classifies the lesion, predicts DNA mutations, and simulates drug responses.
        </p>
        <div className={s.runway}>
          {[['c1','01','AI Engine'],['c2','02','Virtual Biopsy'],['c3','03','Digital Twin']].map(([cls, num, lbl], i, arr) => (
            <div key={cls} className={s.runwayGroup}>
              <div className={s.runwayStop}>
                <div className={`${s.runwayNum} ${s[cls]}`}>{num}</div>
                <div className={s.runwayLbl}>{lbl}</div>
              </div>
              {i < arr.length - 1 && <div className={s.runwayLine} />}
            </div>
          ))}
        </div>
      </div>

      {/* ── Main ── */}
      <div className={s.main}>
        <UploadPanel
          imagePreview={imagePreview}
          form={form}
          onImage={handleImage}
          onClear={clearImage}
          onChange={handleFormChange}
          onRun={handleRun}
          loading={loading}
        />

        <div className={s.rightPanel}>
          {/* Error banner */}
          {error && <div className={s.errorBanner}>{error}</div>}

          {/* Tabs */}
          <div className={s.tabs}>
            {TABS.map((t, i) => (
              <button
                key={i}
                onClick={() => setActiveTab(i)}
                className={`${s.tab} ${activeTab === i ? s.active + ' ' + s[t.cls] : ''}`}
              >
                <span className={s.tabEmoji}>{t.emoji}</span>
                {t.label}
              </button>
            ))}
          </div>

          {/* Phase content */}
          <div className={s.content}>
            {activeTab === 0 && <Phase1 result={results.p1} loading={loading && !results.p1} steps={steps} />}
            {activeTab === 1 && <Phase2 result={results.p2} loading={loading && !!results.p1 && !results.p2} />}
            {activeTab === 2 && <Phase3 result={results.p3} loading={loading && !!results.p2 && !results.p3} />}
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className={s.footer}>
        <span>⚠ Research prototype only · Not for clinical diagnosis · Always consult a licensed dermatologist</span>
        <span className={s.footerStatus}><span className={s.fDot} /> Gemini AI Active</span>
      </footer>
    </div>
  )
}
