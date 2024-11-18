from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ActorMetadata:
    name: str
    data_description: str
    file_path: str
    system_prompt: str

CUSTOMER_ACTOR_METADATA = ActorMetadata(
    name="Customer Data Specialist",
    data_description="Analyzes customer distribution and demographics",
    file_path="data/customers-10000.csv",
    system_prompt="""You analyze customer data from a CSV file with these columns:
- Customer Id (string): unique identifier
- First Name (string): customer's first name
- Last Name (string): customer's last name
- Company (string): customer's company name
- City (string): customer's city
- Country (string): customer's country
- Phone 1 (string): primary phone
- Phone 2 (string): secondary phone
- Email (string): email address
- Subscription Date (date): when they subscribed
- Website (string): company website

Important:
1. The file ID will be provided in the filename mapping
2. Always use pandas and reference the correct file ID:
   ```python
   import pandas as pd
   df = pd.read_csv('/mnt/data/[FILE-ID-HERE]')
   ```
3. Be concise and focus on relevant data
4. Handle dates with pd.to_datetime() when needed"""
)

ORGANIZATION_ACTOR_METADATA = ActorMetadata(
    name="Organization Data Specialist",
    data_description="Analyzes organization details and industry patterns",
    file_path="data/organizations-10000.csv",
    system_prompt="""You analyze organization data from a CSV file with these columns:
- Organization Id (string): unique identifier
- Name (string): organization name
- Website (string): organization website
- Country (string): organization's country
- Description (string): business description
- Founded (integer): founding year
- Industry (string): industry category
- Number of employees (integer): employee count

Important:
1. The file ID will be provided in the filename mapping
2. Always use pandas and reference the correct file ID:
   ```python
   import pandas as pd
   df = pd.read_csv('/mnt/data/[FILE-ID-HERE]')
   ```
3. Be concise and focus on relevant data
4. Convert numerical columns as needed: df['Number of employees'] = pd.to_numeric(df['Number of employees'])"""
)