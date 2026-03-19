import { useState, useEffect } from 'react'
import { checkMLBackend, runPhase1, runPhase2, runPhase3, saveSession, uploadImage } from '../api.js'

export function useAnalysis() {
  const [results, setResults] = useState({ p1: null, p2: null, p3: null })
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState(0)
  const [steps, setSteps] = useState([])
  const [error, setError] = useState(null)
  const [backendOnline, setBackendOnline] = useState(null)

  useEffect(() => {
    // Check if FastAPI model backend is online
    checkMLBackend().then(setBackendOnline)
  }, [])

  const runAnalysis = async (imageFile, imagePreview, form) => {
    try {
      setLoading(true)
      setError(null)
      setResults({ p1: null, p2: null, p3: null })
      setSteps([])
      setActiveTab(0)

      const meta = `Age: ${form.age}, Gender: ${form.gender}, Skin Type: ${form.skinType}, Location: ${form.location}, History: ${form.history}, Notes: ${form.notes}`

      let imageB64 = null
      let imageMime = null
      if (imageFile) {
        setSteps(s => [...s, '🖼️ Processing image...'])
        const buffer = await imageFile.arrayBuffer()
        const base64Str = btoa(new Uint8Array(buffer).reduce((data, byte) => data + String.fromCharCode(byte), ''))
        imageB64 = base64Str
        imageMime = imageFile.type
      }

      setSteps(s => [...s, '🧠 [Phase 1] Analyzing lesion & metadata...'])
      const p1 = await runPhase1(imageFile, imageB64, imageMime, meta)
      setResults(r => ({ ...r, p1 }))
      
      setSteps(s => [...s, '🧬 [Phase 2] Predicting DNA mutations...'])
      setActiveTab(1)
      const p2 = await runPhase2(p1, meta)
      setResults(r => ({ ...r, p2 }))

      setSteps(s => [...s, '👤 [Phase 3] Generating Digital Twin treatment plan...'])
      setActiveTab(2)
      const p3 = await runPhase3(p1, p2, meta)
      setResults(r => ({ ...r, p3 }))

      setSteps(s => [...s, '💾 Saving session...'])
      let imageUrl = null
      if (imageFile) {
        try {
          imageUrl = await uploadImage(imageFile)
        } catch(e) {
          console.warn('Image upload skipped')
        }
      }
      try {
        await saveSession(form, imageUrl, { p1, p2, p3 })
      } catch (e) {
        console.warn('Session save skipped')
      }

      setSteps(s => [...s, '✅ Complete!'])
    } catch (err) {
      console.error(err)
      setError(err.message || 'An error occurred during analysis.')
    } finally {
      setLoading(false)
    }
  }

  return { results, loading, activeTab, setActiveTab, steps, error, backendOnline, runAnalysis }
}
