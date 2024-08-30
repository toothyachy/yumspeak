import os

# mapbox
MAP_API = os.environ.get('MAP_API')

##################  VARIABLES  ##################



##################  CONSTANTS  #####################
# main_category to remove from restaurant_name.csv
CAT_TO_REMOVE = ['Advertising agency', 'Art museum', 'Art studio', 'Association / Organization', 'Beauty salon','Bicycle club', 'Branding agency', 'Business management consultant', 'Candle store', 'Car dealer', 'Car wash', 'Caterer',
'Catering food and drink supplier', "Children's clothing store", 'Chinese Takeaway', 'Chinese language school', 'Clothing store',
'Coaching center', 'Coffee store', 'Condominium complex', 'Consultant', 'Corporate office', 'Curtain and upholstery cleaning service',
'Dance school', 'Delivery Restaurant', 'Designer Clothing Shop', 'Diving center', 'E-commerce service', 'Education center',
'Event management company', 'Fish store', 'Food producer', 'Food products supplier', 'Freight forwarding service', 'Fruit wholesaler'
'Gift shop', 'Gourmet grocery store', 'Gym and Fitness Centre', 'Hawker Stall', 'Hawker center', 'Health and beauty shop',
'Health food store', 'Holding company', 'Hotel', 'Housing development', 'Importer', 'Industrial Real Estate Agency', 'Industrial equipment supplier',
'Information services', 'Interior designer', 'Italian grocery store', 'Japanese confectionery shop', 'Jewelry store', 'Lodging',
'Marketing agency', 'Marketing consultant', 'Maternity store', 'Motorcycle rental agency', 'Movie rental store', 'Music producer',
'Pastry shop', 'Performing arts group', 'Pet Shop', 'Photography service', 'Property management company', 'Publisher', 'Real estate consultant',
'Recording studio', 'Religious organization', 'Restaurant supply store', 'Serviced accommodation', 'Takeaways', 'Tattoo and piercing shop',
'Tattoo shop', 'Tea store', 'Used clothing store', 'Video production service', 'Wellness center', 'Wellness program', 'Wine cellar', "Women's clothing store",
'Bakery', 'Açaí shop', 'Ice cream shop', 'Shop', 'Gift shop', 'Food court', 'Fruit wholesaler', 'Confectionery store',
'Computer support and services']

POSTAL_TO_DISTRICT = {
    '01': '01', '02': '01', '03': '01', '04': '01', '05': '01', '06': '01',
    '07': '02', '08': '02', '09': '04', '10': '04', '11': '05', '12': '05',
    '13': '05', '14': '03', '15': '03', '16': '03', '17': '06', '18': '07',
    '19': '07', '20': '08', '21': '08', '22': '09', '23': '09', '24': '10',
    '25': '10', '26': '10', '27': '10', '28': '11', '29': '11', '30': '11',
    '31': '12', '32': '12', '33': '12', '34': '13', '35': '13', '36': '13',
    '37': '13', '38': '14', '39': '14', '40': '14', '41': '14', '42': '15',
    '43': '15', '44': '15', '45': '15', '46': '16', '47': '16', '48': '16',
    '49': '17', '50': '17', '51': '18', '52': '18', '53': '19', '54': '19',
    '55': '19', '56': '20', '57': '20', '58': '21', '59': '21', '60': '22',
    '61': '22', '62': '22', '63': '22', '64': '22', '65': '23', '66': '23',
    '67': '23', '68': '23', '69': '24', '70': '24', '71': '24', '72': '25',
    '73': '25', '75': '27', '76': '27', '77': '26', '78': '26', '79': '28',
    '80': '28', '81': '17', '82': '19'
}

DISTRICT_TO_REGION = {
    '01': 'City', '02': 'City', '03': 'South', '04': 'South', '05': 'West', '06': 'City',
    '07': 'City', '08': 'Central', '09': 'Central', '10': 'Central', '11': 'Central', '12': 'Central',
    '13': 'East', '14': 'East', '15': 'East', '16': 'East', '17': 'East', '18': 'East',
    '19': 'North', '20': 'North', '21': 'West', '22': 'West', '23': 'West', '24': 'West',
    '25': 'North', '26': 'North', '27': 'North', '28': 'North'
}
################## VALIDATIONS #################
