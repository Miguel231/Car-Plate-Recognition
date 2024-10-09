import requests

def api_car_model(licenseplate_number: str):
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
        <username>{user}</username>
        </CheckSpain>
    </soap:Body>
    </soap:Envelope>"""

    matricula = licenseplate_number 
    user = "MiguelArpa"

    soap_body = soap_body.format(matricula=matricula, user=user)
    response = requests.post(url, data=soap_body, headers=headers)

    print(f"STATE: {response.status_code}")
    print(f"RESPONSE: {response.text}")
