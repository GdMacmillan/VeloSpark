import psycopg2
from datetime import datetime

# conn = psycopg2.connect(dbname='rr_strava_tables', user='gmacmillan', host='localhost')
conn = psycopg2.connect(dbname='strava', user='postgres', host='/tmp')
c = conn.cursor()

c.execute(
    '''SOME QUERY
    ''', {'date': date} # possibly need a datetime, possibly don't
)

conn.commit()
conn.close()
