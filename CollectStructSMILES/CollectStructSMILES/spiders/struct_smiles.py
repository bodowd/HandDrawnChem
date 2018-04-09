import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from CollectStructSMILES.items import CollectStructSMILESItem


class CollectStructSMILESSpider(scrapy.Spider):
    name = 'struct_smiles'
    allowed_domains = ['https://www.ebi.ac.uk/chembl']
    start_urls =['https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL203']

    rules = [
        Rule(LinkExtractor(
            allow = ['https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL*']
        ))
    ]
    # LinkExtractor(allow_domains = ['https://www.ebi.ac.uk/chembl/target/inspect'])

    def parse(self, response):
        links = LinkExtractor().extract_links(response)
        # selector_list = response.css('td')

        for link in links:

            item = CollectStructSMILESItem()
            item['image'] = response.css('td#molecule-image-cell').xpath('img').extract()
            # returns a list. element in index 8 contains canonical smiles
            item['smiles'] = response.css('td.alternaterowcolour::text').extract()[8]
            item['name'] = response.css('td.alternaterowcolour::text').extract()[1]
            yield item
