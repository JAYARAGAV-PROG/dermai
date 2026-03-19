import { createClient } from '@supabase/supabase-js'
const URL = import.meta.env.VITE_SUPABASE_URL
const KEY = import.meta.env.VITE_SUPABASE_ANON_KEY
const dummy = { from:()=>({ insert:()=>({ select:()=>({ single:()=>({data:null,error:new Error('Not configured')}) }) }), select:()=>({ order:()=>({ limit:()=>({data:[],error:null}) }) }) }), storage:{ from:()=>({ upload:()=>({data:null,error:new Error('Not configured')}), getPublicUrl:()=>({data:{publicUrl:null}}) }) } }
export const supabase = (URL && KEY) ? createClient(URL, KEY) : dummy
