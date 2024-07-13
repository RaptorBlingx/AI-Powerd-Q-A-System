import requests

url = 'http://127.0.0.1:5000/add_document'
document = {
    "title": "Türkiye",
    "content": "Türkiye'nin çeşitli bölgelerinde yerel yemek kültürleri oldukça zengindir. Özellikle İstanbul, geleneksel lezzetler ve sokak yemekleriyle ünlüdür. Bu şehirde, simit ve döner gibi sokak lezzetleri, hem yerel halkın hem de turistlerin vazgeçilmezleri arasında yer alır. Ayrıca, Anadolu mutfağından esinlenilmiş birçok farklı tat da İstanbul'da bulunabilir",
    "reference": "test_document_007",
    "language": "tr"
}

response = requests.post(url, json=document)
print(response.json())
