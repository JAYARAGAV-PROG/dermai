import s from './App.module.css'

export default function UploadPanel({ imagePreview, form, onImage, onClear, onChange, onRun, loading }) {
  return (
    <div className={s.leftPanel}>
      {/* ── Image Upload ── */}
      <div className={s.uploadBox}>
        {imagePreview ? (
          <div className={s.preview}>
            <img src={imagePreview} alt="Lesion preview" />
            <button className={s.clearBtn} onClick={onClear} disabled={loading}>×</button>
          </div>
        ) : (
          <div className={s.dropzone}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={e => onImage(e.target.files[0])} 
              disabled={loading} 
            />
            <div className={s.dropIcon}>📸</div>
            <p>Drag & Drop lesion photo</p>
            <span>or click to browse</span>
          </div>
        )}
      </div>

      {/* ── Patient Metadata Form ── */}
      <div className={s.form}>
        <div className={s.formGroup}>
          <label>Age</label>
          <input type="number" placeholder="e.g. 45" value={form.age} onChange={e => onChange('age', e.target.value)} disabled={loading} />
        </div>
        <div className={s.formGroup}>
          <label>Gender</label>
          <select value={form.gender} onChange={e => onChange('gender', e.target.value)} disabled={loading}>
            <option value="">Select...</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <div className={s.formGroup}>
          <label>Fitzpatrick Skin Type</label>
          <select value={form.skinType} onChange={e => onChange('skinType', e.target.value)} disabled={loading}>
            <option value="">Select...</option>
            <option value="I">Type I (Pale, burns easily)</option>
            <option value="II">Type II</option>
            <option value="III">Type III</option>
            <option value="IV">Type IV</option>
            <option value="V">Type V</option>
            <option value="VI">Type VI (Deeply pigmented)</option>
          </select>
        </div>
        <div className={s.formGroup}>
          <label>Lesion Location</label>
          <input type="text" placeholder="e.g. Upper back" value={form.location} onChange={e => onChange('location', e.target.value)} disabled={loading} />
        </div>
        <div className={s.formGroup} style={{ gridColumn: '1 / -1' }}>
          <label>Medical History</label>
          <input type="text" placeholder="e.g. Previous sunburns, family history of melanoma" value={form.history} onChange={e => onChange('history', e.target.value)} disabled={loading} />
        </div>
        <div className={s.formGroup} style={{ gridColumn: '1 / -1' }}>
          <label>Clinical Notes</label>
          <textarea placeholder="Any additional observations..." value={form.notes} onChange={e => onChange('notes', e.target.value)} disabled={loading} rows={2} />
        </div>
      </div>

      {/* ── Run Action ── */}
      <button 
        className={s.runBtn} 
        disabled={loading || !imagePreview || !form.age || !form.gender} 
        onClick={onRun}
      >
        {loading ? <span className={s.spinnerSmall} /> : 'Run Full Analysis'}
      </button>
    </div>
  )
}
