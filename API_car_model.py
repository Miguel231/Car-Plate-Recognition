import requests

url = "http://www.regcheck.org.uk/api/reg.asmx"

headers = {
    'Content-Type': 'text/xml; charset=utf-8',
    'SOAPAction': 'http://regcheck.org.uk/CheckSpain'
}

#SOAP MESSAGE
soap_body = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <CheckSpain xmlns="http://regcheck.org.uk">
      <RegistrationNumber>{matricula}</RegistrationNumber>
      <username>{usuario}</username>
    </CheckSpain>
  </soap:Body>
</soap:Envelope>"""

matricula = "9660FTN"  
usuario = "MiguelArpa"

soap_body = soap_body.format(matricula=matricula, usuario=usuario)
response = requests.post(url, data=soap_body, headers=headers)

#print(f"STATE: {response.status_code}")
print(f"RESPONSE: {response.text}")
