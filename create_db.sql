CREATEDB rr_strava_tables

CONNECT rr_strava_tables

CREATE TABLE users (
  user_id           int,            -- strava user id
  resource_state    int,            -- strava user resource availability
  firstname         varchar(80),
  lastname          varchar(80),
  profile_medium    text,           -- profile pic medium
  profile           text,           -- profile pic large
  city              varchar(80),    -- user city
  state             varchar(80),    -- user states
  country           varchar(80),    -- user country
  sex               varchar(10),
  friend            varchar(10),    -- the authenticated athlete’s following status of this athlete
  follower          varchar(10),    -- this athlete’s following status of the authenticated athlete
  premium           boolean,        -- whether user's account is premium or not
  created_at        timestamp,      -- when users account was created
  updated_at        timestamp       -- last time users account was updated
                    );
