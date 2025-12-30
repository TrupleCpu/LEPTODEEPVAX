from Bio import Entrez, SeqIO
import time

class NCBIGateway:
    """
    Automated Biological Data Acquisition System.
    Targets specific proteomic records from the NCBI database 
    to serve as the 'Raw Material' for retrosynthetic analysis.
    """
    def __init__(self, email):
        Entrez.email = email
        self.db = "protein"

    def search_and_fetch(self, query, max_results=50):
        print(f"[FETCH] Querying NCBI for: {query}")
        try:
            # Search for IDs
            handle = Entrez.esearch(db=self.db, term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record.get("IdList", [])
            if not id_list:
                print("[ERROR] No protein records found.")
                return []

            # Fetch Fasta sequences
            print(f"[FETCH] Downloading {len(id_list)} sequences...")
            handle = Entrez.efetch(db=self.db, id=id_list, rettype="fasta", retmode="text")
            records = list(SeqIO.parse(handle, "fasta"))
            handle.close()

            # Parse into structured list
            parsed_data = []
            for r in records:
                parsed_data.append({
                    "id": r.id,
                    "desc": r.description,
                    "seq": str(r.seq)
                })
            return parsed_data

        except Exception as e:
            print(f"[CRITICAL] NCBI Gateway failed: {e}")
            return []