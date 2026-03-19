import s from './Phase.module.css'
export default function Phase3({ result, loading }) {
  if (loading && !result) return <div className={s.spinner} style={{paddingTop:80}}><div className={s.ring} style={{borderTopColor:'var(--accent3)'}}/><p className={s.spinText}>Building Digital Twin…<br/>Simulating drug responses in-silico</p></div>
  if (!result) return <div className={s.empty}><div className={s.emptyIcon}>👤</div><h3>Digital Twin Pending</h3><p>Complete Phase 1 + 2 to generate<br/>personalized treatment simulation</p></div>
  return (
    <div className={s.wrap}>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(255,107,53,.12)'}}>💊</span><div><b>Treatment Simulation</b><span>Digital Twin · In-Silico Drug Testing</span></div></div>
        <div className={s.drugGrid}>
          {result.drugs.map((d,i)=>(
            <div key={i} className={`${s.drugCard} ${d.recommended?s.recommended:''}`}>
              {d.recommended && <span className={s.recBadge}>✓ Top Pick</span>}
              <div className={s.drugName}>{d.name}</div>
              <div className={s.drugClass}>{d.class}</div>
              <div className={s.eTrack}><div className={s.eFill} style={{width:d.efficacy+'%'}}/></div>
              <div className={s.eLabel}>Efficacy: {d.efficacy}%</div>
            </div>
          ))}
        </div>
      </div>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(255,107,53,.12)'}}>🖼️</span><div><b>3-Month Visual Simulation</b></div></div>
        <div className={s.simGrid}>
          <div className={s.simPanel}><div className={s.simLabel}>Day 0 — Current</div><div className={`${s.simVisual} ${s.before}`}><div className={s.moleBefore}/></div><p className={s.simNote}>{result.beforeDesc}</p></div>
          <div className={s.simPanel}><div className={s.simLabel}>Week 12 — Projected</div><div className={`${s.simVisual} ${s.after}`}><div className={s.moleAfter}/></div><p className={s.simNote}>{result.afterDesc}</p></div>
        </div>
      </div>
      <div className={s.planCard}>
        <div className={s.planTitle}>📋 Dr. AI Clinical Brief</div>
        <div className={s.planSub}>Generated for physician handoff</div>
        <pre className={s.planText}>{result.plan}</pre>
      </div>
    </div>
  )
}
