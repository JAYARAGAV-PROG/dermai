/**
 * DermAI API Layer — Gemini + FastAPI + Supabase
 */
import { supabase } from './supabase.js'

const GEMINI_KEY = import.meta.env.VITE_GEMINI_API_KEY
const ML_BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_KEY}`

// ── Gemini ────────────────────────────────────────────────────────────────────
export async function callGemini(prompt, systemPrompt, imageParts = null) {
  if (!GEMINI_KEY) throw new Error('MISSING_GEMINI_KEY')
  const parts = []
  if (imageParts) parts.push({ inline_data: { mime_type: imageParts.mimeType, data: imageParts.data } })
  parts.push({ text: prompt })
  const res = await fetch(GEMINI_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: systemPrompt }] },
      contents: [{ role: 'user', parts }],
      generationConfig: { temperature: 0.3, maxOutputTokens: 4096 }
    })
  })
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e?.error?.message || `Gemini error ${res.status}`) }
  const data = await res.json()
  return data.candidates?.[0]?.content?.parts?.[0]?.text || ''
}

export function parseJSON(raw) {
  try {
    return JSON.parse(raw.replace(/```json|```/g, '').trim())
  } catch (e) {
    console.error('JSON parse failed. Raw response:', raw)
    throw new Error('AI response was incomplete — please try again')
  }
}

// ── ML Backend ────────────────────────────────────────────────────────────────
export async function getPredictionFromModel(imageFile) {
  const fd = new FormData(); fd.append('file', imageFile)
  const res = await fetch(`${ML_BACKEND}/predict`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(`ML backend error: ${res.status}`)
  return res.json()
}

export async function checkMLBackend() {
  try { const r = await fetch(`${ML_BACKEND}/health`, { signal: AbortSignal.timeout(3000) }); return r.ok }
  catch { return false }
}

// ── Phase 1 ───────────────────────────────────────────────────────────────────
export async function runPhase1(imageFile, imageB64, imageMime, meta) {
  let mlResult = null
  if (imageFile) { try { mlResult = await getPredictionFromModel(imageFile) } catch(e) { console.warn('ML offline:', e.message) } }

  const system = `You are a dermatology AI engine performing multi-modal skin lesion analysis.
${mlResult ? 'You are given REAL probability scores from EfficientNet-B4 trained on HAM10000. Use these exact numbers.' : 'Analyze the provided lesion image and patient metadata.'}
Return ONLY valid JSON. No markdown. Schema:
{"classification":"string","riskScore":number,"riskLevel":"low|medium|high","riskRationale":"string","metadataInsights":"string"}`

  const prompt = mlResult
    ? `EfficientNet-B4 scores:\n${mlResult.all_probabilities.map(p=>`  ${p.name}: ${p.probability}%${p.malignant?' [MALIGNANT]':''}`).join('\n')}\nMalignancy risk: ${mlResult.malignancy_risk}%\nRisk level: ${mlResult.risk_level}\nPatient: ${meta}\nProvide clinical narrative. JSON only.`
    : `Analyze skin lesion. Patient: ${meta}. JSON only.`

  const raw = await callGemini(prompt, system, imageB64 ? { mimeType: imageMime||'image/jpeg', data: imageB64 } : null)
  const result = parseJSON(raw)
  result._ml = mlResult ? { used:true, demoMode:mlResult.demo_mode, topClass:mlResult.top_class_name } : { used:false }
  return result
}

// ── Phase 2 ───────────────────────────────────────────────────────────────────
export async function runPhase2(phase1Result, meta) {
  const system = `You are a computational genomics AI predicting DNA mutations from dermoscopic findings.
Return ONLY valid JSON. No markdown. Schema:
{"mutations":[{"name":"BRAF V600E","detected":bool},{"name":"NRAS Q61R","detected":bool},{"name":"KIT D816V","detected":bool},{"name":"CDKN2A","detected":bool},{"name":"TP53","detected":bool},{"name":"PTEN loss","detected":bool},{"name":"MC1R variant","detected":bool},{"name":"NF1","detected":bool}],"analysis":"string","pathways":"string"}`

  const raw = await callGemini(
    `Classification: ${phase1Result.classification}\nRisk: ${phase1Result.riskScore}%\nPatient: ${meta}\nPredict DNA mutations. JSON only.`,
    system
  )
  return parseJSON(raw)
}

// ── Phase 3 ───────────────────────────────────────────────────────────────────
export async function runPhase3(p1, p2, meta) {
  const detected = p2.mutations.filter(m=>m.detected).map(m=>m.name).join(', ') || 'None'
  const system = `You are a precision oncology AI building a Digital Twin for in-silico drug simulation.
Return ONLY valid JSON. No markdown. Schema:
{"drugs":[{"name":"string","class":"string","efficacy":number,"recommended":bool},{"name":"string","class":"string","efficacy":number,"recommended":false},{"name":"string","class":"string","efficacy":number,"recommended":false},{"name":"string","class":"string","efficacy":number,"recommended":false}],"beforeDesc":"string","afterDesc":"string","plan":"string"}
Exactly ONE drug recommended:true. Efficacy range 45-95. Real oncology drug names. Plan = 150-200 word physician brief.`

  const raw = await callGemini(
    `Patient: ${meta}\nClassification: ${p1.classification}\nRisk: ${p1.riskScore}%\nDetected mutations: ${detected}\nSimulate drug responses. JSON only.`,
    system
  )
  return parseJSON(raw)
}

// ── Supabase: Save session ────────────────────────────────────────────────────
export async function saveSession(formData, imageUrl, results) {
  try {
    const { data: session, error } = await supabase.from('sessions').insert({
      patient_age: parseInt(formData.age), patient_gender: formData.gender,
      skin_type: formData.skinType||null, lesion_location: formData.location||null,
      medical_history: formData.history||null, clinical_notes: formData.notes||null,
      image_url: imageUrl||null,
    }).select().single()
    if (error) throw error
    for (const [phase, data] of [[1,results.p1],[2,results.p2],[3,results.p3]]) {
      if (data) await supabase.from('results').insert({ session_id:session.id, phase, result_json:data, risk_score:data.riskScore||null, risk_level:data.riskLevel||null })
    }
    return session.id
  } catch(e) { console.error('Save error:', e.message); return null }
}

// ── Supabase: Upload image ────────────────────────────────────────────────────
export async function uploadImage(file) {
  try {
    const filename = `${Date.now()}_${Math.random().toString(36).slice(2)}.${file.name.split('.').pop()}`
    const { error } = await supabase.storage.from('lesion-images').upload(filename, file)
    if (error) throw error
    return supabase.storage.from('lesion-images').getPublicUrl(filename).data.publicUrl
  } catch(e) { console.error('Upload error:', e.message); return null }
}

// ── Supabase: Get history ─────────────────────────────────────────────────────
export async function getSessionHistory() {
  const { data, error } = await supabase.from('sessions').select('*,results(*)').order('created_at',{ascending:false}).limit(20)
  if (error) { console.error(error); return [] }
  return data
}
