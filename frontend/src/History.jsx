import { useState, useEffect } from 'react'
import { getSessionHistory } from './api.js'
import s from './History.module.css'

export default function History({ onBack }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  useEffect(() => { getSessionHistory().then(d => { setSessions(d); setLoading(false) }) }, [])
  return (
    <div className={s.page}>
      <div className={s.header}>
        <button className={s.back} onClick={onBack}>← Back</button>
        <h2>Session History</h2>
        <span className={s.count}>{sessions.length} records</span>
      </div>
      {loading && <div className={s.loading}>Loading sessions…</div>}
      {!loading && sessions.length === 0 && <div className={s.empty}><div>📋</div><p>No sessions yet. Run your first analysis to see history here.</p></div>}
      <div className={s.list}>
        {sessions.map(session => {
          const p1 = session.results?.find(r => r.phase === 1)
          return (
            <div className={s.card} key={session.id}>
              <div className={s.cardTop}>
                <div><div className={s.cardId}>#{session.id.slice(0,8)}</div><div className={s.cardDate}>{new Date(session.created_at).toLocaleString()}</div></div>
                {p1 && <div className={`${s.risk} ${s[p1.risk_level]}`}>{p1.risk_score}% risk</div>}
              </div>
              <div className={s.meta}>
                <span>{session.patient_age}yo {session.patient_gender}</span>
                {session.skin_type && <span>{session.skin_type}</span>}
                {session.lesion_location && <span>{session.lesion_location}</span>}
              </div>
              {p1?.result_json?.classification && <p className={s.classification}>{p1.result_json.classification}</p>}
              {session.image_url && <img src={session.image_url} alt="lesion" className={s.thumb} />}
            </div>
          )
        })}
      </div>
    </div>
  )
}
