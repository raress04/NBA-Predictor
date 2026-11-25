"""etl/odds_fetcher.py::parse_player_props: an over and an under are only paired at the same line."""
from etl.odds_fetcher import parse_player_props


def _o(player, side, point, price):
    return {'name': side, 'description': player, 'point': point, 'price': price}


def _book(key, outcomes, market='player_points'):
    return {'key': key, 'markets': [{'key': market, 'outcomes': outcomes}]}


def test_single_line_both_sides():
    data = {'bookmakers': [_book('pinnacle', [_o('LeBron James', 'Over', 16.5, 1.90),
                                              _o('LeBron James', 'Under', 16.5, 1.95)])]}
    assert parse_player_props(data)['LeBron James']['points'] == {
        'line': 16.5, 'over_odds': 1.90, 'under_odds': 1.95, 'book': 'pinnacle'}


def test_alternate_lines_are_not_mixed():
    # Order that the old player-only grouping got wrong: it paired the 20.5 over with the 16.5 under.
    outs = [_o('LeBron James', 'Over', 16.5, 1.90), _o('LeBron James', 'Over', 20.5, 3.10),
            _o('LeBron James', 'Under', 16.5, 1.95), _o('LeBron James', 'Under', 20.5, 1.36)]
    for book in ('pinnacle', 'betonlineag'):            # preferred and non-preferred path
        p = parse_player_props({'bookmakers': [_book(book, outs)]})['LeBron James']['points']
        assert (p['line'], p['over_odds'], p['under_odds']) == (16.5, 1.90, 1.95)


def test_missing_side_filled_only_at_same_line():
    data = {'bookmakers': [
        _book('betonlineag', [_o('A', 'Over', 10.5, 1.80)]),
        _book('bovada', [_o('A', 'Under', 11.5, 2.00)]),          # different line: not used
        _book('fanduel', [_o('A', 'Under', 10.5, 1.98)]),         # same line: used
    ]}
    p = parse_player_props(data)['A']['points']
    assert (p['line'], p['over_odds'], p['under_odds'], p['book']) == (10.5, 1.80, 1.98, 'betonlineag')


def test_missing_side_stays_none_without_same_line():
    data = {'bookmakers': [_book('betonlineag', [_o('A', 'Over', 10.5, 1.80)]),
                           _book('bovada', [_o('A', 'Under', 11.5, 2.00)])]}
    assert parse_player_props(data)['A']['points']['under_odds'] is None


def test_preferred_book_replaces_other_book():
    data = {'bookmakers': [
        _book('betonlineag', [_o('A', 'Over', 10.5, 1.80), _o('A', 'Under', 10.5, 2.00)]),
        _book('pinnacle', [_o('A', 'Over', 11.5, 1.91), _o('A', 'Under', 11.5, 1.93)]),
    ]}
    assert parse_player_props(data)['A']['points'] == {
        'line': 11.5, 'over_odds': 1.91, 'under_odds': 1.93, 'book': 'pinnacle'}


def test_preferred_book_with_under_only_does_not_replace():
    data = {'bookmakers': [
        _book('betonlineag', [_o('A', 'Over', 10.5, 1.80), _o('A', 'Under', 10.5, 2.00)]),
        _book('pinnacle', [_o('A', 'Under', 11.5, 1.93)]),
    ]}
    assert parse_player_props(data)['A']['points']['book'] == 'betonlineag'


def test_categories_and_unknown_markets():
    data = {'bookmakers': [{'key': 'pinnacle', 'markets': [
        {'key': 'player_rebounds', 'outcomes': [_o('A', 'Over', 5.5, 1.9), _o('A', 'Under', 5.5, 1.9)]},
        {'key': 'player_threes', 'outcomes': [_o('A', 'Over', 1.5, 1.9), _o('A', 'Under', 1.5, 1.9)]},
    ]}]}
    assert set(parse_player_props(data)['A']) == {'rebounds'}
