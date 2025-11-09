create table if not exists threads (
  id serial primary key,
  platform varchar(16) not null,
  channel_id text not null,
  thread_ts text,
  source_url text,
  created_at timestamptz default now()
);

create table if not exists actions (
  id serial primary key,
  thread_id int references threads(id),
  kind varchar(32) not null,           -- create, update, comment, approval_request
  payload jsonb not null,
  status varchar(16) not null default 'queued', -- queued, done, failed
  created_at timestamptz default now()
);

create table if not exists issue_links (
  id serial primary key,
  thread_id int references threads(id),
  issue_key text not null,
  last_status text,
  created_at timestamptz default now(),
  unique (thread_id, issue_key)
);
