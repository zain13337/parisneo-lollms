import sqlite3

class SkillsLibrary:
        
    def __init__(self, db_path):
        self.db_path =db_path
        self._initialize_db()

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills_library (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                category TEXT,
                title TEXT,
                content TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_info (
                version INTEGER
            )
        """)
        cursor.execute("SELECT version FROM db_info")
        version = cursor.fetchone()
        if version is None:
            cursor.execute("INSERT INTO db_info (version) VALUES (1)")
            conn.commit()
            cursor.close()
            conn.close()
        else:
            cursor.close()
            conn.close()
            self._migrate_db(version[0])

    def _migrate_db(self, version):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        # Perform migrations based on the current version
        # For example, if the current version is 1 and the latest version is 2:
        if version < 2:
            cursor.execute("ALTER TABLE skills_library ADD COLUMN new_column TEXT")
            cursor.execute("UPDATE db_info SET version = 2")
            conn.commit()
        cursor.close()
        conn.close()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills_library (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                category TEXT,
                title TEXT,
                content TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

    def add_entry(self, version, category, title, content):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("""
            INSERT INTO skills_library (version, category, title, content) 
            VALUES (?, ?, ?, ?)
        """, (version, category, title, content))
        conn.commit()
        cursor.close()
        conn.close()

    def list_entries(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("SELECT * FROM skills_library")
        res = cursor.fetchall()
        cursor.close()
        conn.close()
        return res
    
    def query_entry(self, text):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("""
            SELECT * FROM skills_library 
            WHERE category LIKE ? OR title LIKE ? OR content LIKE ?
        """, (f'%{text}%', f'%{text}%', f'%{text}%'))
        res= cursor.fetchall()
        cursor.close()
        conn.close()
        return res
    
    def remove_entry(self, id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()        
        cursor.execute("DELETE FROM skills_library WHERE id = ?", (id,))
        conn.commit()
        cursor.close()
        conn.close()

    def export_entries(self, file_path):
        with open(file_path, 'w') as f:
            for entry in self.list_entries():
                f.write(f'{entry}\n')

    def import_entries(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                entry = line.strip().split(',')
                self.add_entry(*entry)

    def fuse_with_another_db(self, other_db_path):
        other_conn = sqlite3.connect(other_db_path)
        other_cursor = other_conn.cursor()
        other_cursor.execute("SELECT * FROM skills_library")
        for row in other_cursor.fetchall():
            self.add_entry(*row[1:])  # skip the id column
