import s from './Phase.module.css'
export default function Phase2({ result, loading }) {
  if (loading && !result) return <div className={s.spinner} style={{paddingTop:80}}><div className={s.ring} style={{borderTopColor:'var(--accent2)'}}/><p className={s.spinText}>Virtual Biopsy in progress…<br/>Predicting molecular markers</p></div>
  if (!result) return <div className={s.empty}><div className={s.emptyIcon}>🧬</div><h3>Virtual Biopsy Awaiting</h3><p>Complete Phase 1 to unlock<br/>non-invasive genomic prediction</p></div>
  return (
    <div className={s.wrap}>
      <div className={s.card} style={{borderColor:'var(--accent2)',boxShadow:'0 0 30px rgba(123,47,255,.15)'}}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(123,47,255,.12)'}}>🧬</span><div><b>Predicted Genomic Mutations</b><span>Virtual Biopsy · No Scalpel Required</span></div></div>
        <div className={s.chips}>{result.mutations.map((m,i)=><span key={i} className={`${s.chip} ${m.detected?s.detected:s.negative}`}>{m.detected?'✓ ':'✗ '}{m.name}</span>)}</div>
      </div>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(123,47,255,.12)'}}>🔬</span><div><b>Molecular Analysis</b></div></div>
        <p className={s.mono}>{result.analysis}</p>
      </div>
      <div className={s.card}>
        <div className={s.cardHead}><span className={s.icon} style={{background:'rgba(123,47,255,.12)'}}>⚡</span><div><b>Signaling Pathway Disruption</b></div></div>
        <p className={s.mono}>{result.pathways}</p>
      </div>
    </div>
  )
}
