-- ═══════════════════════════════════════════════════════════════
-- DermAI — Supabase Database Schema
-- ═══════════════════════════════════════════════════════════════
-- HOW TO USE:
-- 1. Go to https://supabase.com → your project
-- 2. Click "SQL Editor" in the left sidebar
-- 3. Paste this entire file and click "Run"
-- ═══════════════════════════════════════════════════════════════

-- Enable UUID generation
create extension if not exists "uuid-ossp";

-- ── TABLE: sessions ──────────────────────────────────────────────
-- One row per analysis run
create table if not exists sessions (
  id               uuid primary key default uuid_generate_v4(),
  created_at       timestamptz default now(),
  patient_age      integer,
  patient_gender   text,
  skin_type        text,
  lesion_location  text,
  medical_history  text,
  clinical_notes   text,
  image_url        text        -- public URL from Supabase Storage
);

-- ── TABLE: results ───────────────────────────────────────────────
-- One row per phase (1, 2, 3) per session
create table if not exists results (
  id           uuid primary key default uuid_generate_v4(),
  created_at   timestamptz default now(),
  session_id   uuid references sessions(id) on delete cascade,
  phase        integer not null,        -- 1, 2, or 3
  result_json  jsonb,                   -- full AI output
  risk_score   numeric,                 -- Phase 1 only
  risk_level   text                     -- 'low' | 'medium' | 'high'
);

-- ── INDEXES ──────────────────────────────────────────────────────
create index if not exists idx_results_session on results(session_id);
create index if not exists idx_sessions_created on sessions(created_at desc);

-- ── ROW LEVEL SECURITY ───────────────────────────────────────────
-- For now: public read/write (fine for academic demo)
-- Later: add auth and restrict to logged-in users only
alter table sessions enable row level security;
alter table results  enable row level security;

create policy "Allow all for now" on sessions for all using (true) with check (true);
create policy "Allow all for now" on results  for all using (true) with check (true);

-- ── STORAGE BUCKET ───────────────────────────────────────────────
-- Run this separately in the SQL editor:
insert into storage.buckets (id, name, public)
values ('lesion-images', 'lesion-images', true)
on conflict do nothing;

-- Allow public uploads to the bucket
create policy "Public uploads" on storage.objects
  for insert with check (bucket_id = 'lesion-images');

create policy "Public reads" on storage.objects
  for select using (bucket_id = 'lesion-images');

-- ── DONE ─────────────────────────────────────────────────────────
-- Your database is ready.
-- Next: copy your Supabase URL + anon key into frontend/.env
