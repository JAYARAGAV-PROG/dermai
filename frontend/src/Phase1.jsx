import s from './Phase.module.css'
export default function Phase1({ result, loading, steps }) {
  if (loading && !result) return (
    <div className={s.wrap}>
      <div className={s.stepBox}>{steps.map((st,i)=><div className={s.step} key={i}><div className={`${s.dot} ${s[st.status]}`}/><span className={`${s.stepLabel} ${s[st.status]}`}>{st.label}</span></div>)}</div>
      <div className={s.spinner}><div className={s.ring} style={{borderTopColor:'var(--accent)'}}/><p className={s.spinText}>AI Engine analyzing…<br/>Fusing image + metadata</p></div>
    </div>
  )
  if (!result) return <div className={s.empty}><div className={s.emptyIcon}>🧠</div><h3>AI Engine Ready</h3><p>Upload a lesion photo + patient info<br/>to activate multi-modal analysis</p></div>
  return (
    <div className={s.wrap}>
      {result._ml?.used && <div className={s.mlBadge}>⚡ EfficientNet-B4 · {result._ml.topClass} · {result._ml.demoMode?'Demo mode':'Live model'}</div>}
      <div className={s.card} style={{borderColor:'var(--accent)',boxShadow:'var(--glow)'}}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(0,229,255,.12)'}}>🔍</span><div><b>Multi-Modal Classification</b><span>Engine Output · Phase 1</span></div></div>
        <p className={s.mono}>{result.classification}</p>
      </div>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(0,229,255,.12)'}}>📊</span><div><b>Risk Assessment</b></div></div>
        <div className={s.riskRow}><span className={s.mono}>Malignancy Risk</span><span className={s.mono}>{result.riskScore}%</span></div>
        <div className={s.track}><div className={`${s.fill} ${s[result.riskLevel]}`} style={{width:result.riskScore+'%'}}/></div>
        <p className={`${s.mono} ${s.mt}`}>{result.riskRationale}</p>
      </div>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(0,229,255,.12)'}}>🎯</span><div><b>Metadata Filter Insights</b></div></div>
        <p className={s.mono}>{result.metadataInsights}</p>
      </div>
    </div>
  )
}
