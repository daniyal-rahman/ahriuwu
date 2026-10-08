// 26.19 inventory and shop rules (lanerl_jax/modern/items/inventory.py) for one champion's 7-slot inventory
// (``owned_counts``/``inventory_stats``/``inventory_from_ids`` take the (C, 7) inventory). Catalog arrays come from
// native/python/consts/econ_inventory.py.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../marshal.hpp"
#include "../stats.hpp"
#include "econ.hpp"

namespace lanesim::econ {

using namespace champ;

namespace {

constexpr int MAX_RECIPE_NODES = 16, STEALTH_WARD = 3340;
constexpr float SHOP_RADIUS = 1000.f;
constexpr float SHOP_CENTER[2][2] = {{412.9f, 416.2f}, {14297.2f, 14388.3f}};

struct Cat {
    std::vector<float> item_id, stats, multiplicative, total, sell_value, can_be_sold, in_store, max_stack, groups,
        group_max, group_purchase_cd, trinket, required_level, ranged_only, blocked, required_buff, node_item,
        node_parent, node_total;
    int n_items, n_groups, n_stats;
    Cat() {
        auto t = [](const char* n) { return data::table(std::string("econ.inventory.") + n); };
        item_id = t("item_id"), stats = t("stats"), multiplicative = t("multiplicative"), total = t("total");
        sell_value = t("sell_value"), can_be_sold = t("can_be_sold"), in_store = t("in_store");
        max_stack = t("max_stack"), groups = t("groups"), group_max = t("group_max");
        group_purchase_cd = t("group_purchase_cd"), trinket = t("trinket"), required_level = t("required_level");
        ranged_only = t("ranged_only"), blocked = t("blocked"), required_buff = t("required_buff");
        node_item = t("node_item"), node_parent = t("node_parent"), node_total = t("node_total");
        n_items = (int)item_id.size(), n_groups = (int)group_max.size(), n_stats = (int)multiplicative.size();
    }
    int row(int id) const {
        for (int r = 0; r < n_items; ++r)
            if ((int)item_id[r] == id) return r;
        throw std::out_of_range("econ: item id not in the catalog");
    }
    bool group(int r, int g) const { return groups[(size_t)r * n_groups + g] != 0.f; }
};
const Cat& K() {
    static const Cat k;
    return k;
}
int clip_row(int r) { return clampi(r, 0, K().n_items - 1); }   // JAX gathers clamp out-of-range rows

}  // namespace

// inventory.inventory_from_ids (host helper; ``ids`` (C, k), 0 = none)
Inventory inventory_from_ids(const Arr<int32_t>& ids, int k, bool trinket) {
    const Cat& cat = K();
    size_t c = k > 0 ? ids.size() / k : 0;
    Inventory inv;
    inv.item.assign(c * N_SLOTS, EMPTY), inv.stack.assign(c * N_SLOTS, 0);
    for (size_t h = 0; h < c; ++h) {
        int32_t* item = &inv.item[h * N_SLOTS];
        int32_t* stack = &inv.stack[h * N_SLOTS];
        if (trinket) item[TRINKET_SLOT] = cat.row(STEALTH_WARD), stack[TRINKET_SLOT] = 1;
        int slot = 0;
        for (int q = 0; q < k; ++q) {
            int iid = ids[h * k + q];
            if (iid == 0) continue;
            int row = cat.row(iid);
            if (cat.trinket[row] != 0.f) {
                item[TRINKET_SLOT] = row, stack[TRINKET_SLOT] = 1;
                continue;
            }
            int ms = (int)cat.max_stack[row], same = -1;
            for (int s = 0; s < TRINKET_SLOT && same < 0; ++s)
                if (item[s] == row && stack[s] < ms) same = s;
            if (ms > 1 && same >= 0) {
                stack[same] += 1;
                continue;
            }
            if (slot >= TRINKET_SLOT) throw std::invalid_argument("more than six inventory items");
            item[slot] = row, stack[slot] = 1;
            ++slot;
        }
    }
    return inv;
}

// inventory.owned_counts: (..., I) units owned of each catalog row (stacks counted)
Arr<int32_t> owned_counts(const Inventory& inv) {
    int ni = K().n_items;
    size_t c = inv.item.size() / N_SLOTS;
    Arr<int32_t> out(c * ni, 0);
    for (size_t h = 0; h < c; ++h)
        for (int s = 0; s < N_SLOTS; ++s) {
            int r = inv.item[h * N_SLOTS + s];
            if (r >= 0 && r < ni) out[h * ni + r] += inv.stack[h * N_SLOTS + s];
        }
    return out;
}

// inventory.inventory_stats: static item stats; a stacked slot counts once
ItemStats inventory_stats(const Inventory& inv) {
    const Cat& cat = K();
    size_t c = inv.item.size() / N_SLOTS;
    ItemStats out = stats::zero(c);
    auto fields = stats::fields(out);
    for (size_t h = 0; h < c; ++h)
        for (int f = 0; f < cat.n_stats; ++f) {
            float add = 0.f, prod = 1.f;
            for (int s = 0; s < N_SLOTS; ++s) {
                int it = inv.item[h * N_SLOTS + s];
                bool present = it >= 0 && inv.stack[h * N_SLOTS + s] > 0;
                float v = present ? cat.stats[(size_t)clip_row(it) * cat.n_stats + f] : 0.f;
                add = add + v, prod = prod * (1.f - v);
            }
            (*fields[f])[h] = cat.multiplicative[f] != 0.f ? 1.f - prod : add;
        }
    return out;
}

// inventory.in_shop_area
bool in_shop_area(float x, float z, int team, bool dead) {
    const float* centre = SHOP_CENTER[clampi(team, 0, 1)];
    float d2 = sq(x - centre[0]) + sq(z - centre[1]);
    return dead || d2 <= SHOP_RADIUS * SHOP_RADIUS;
}

// inventory.buy: a failed purchase changes nothing
ShopResult buy(const Inventory& inv, float gold, int row_in, bool can_shop, int level, bool is_ranged, float now,
               const Arr<float>& group_cd_in, const Arr<uint8_t>& buff_currency) {
    const Cat& cat = K();
    const int ng = cat.n_groups;
    const int row = clip_row(row_in);
    Arr<float> group_cd_until = group_cd_in.size() ? group_cd_in : Arr<float>(ng, 0.f);
    const int32_t* items = inv.item.data();
    const int32_t* stacks = inv.stack.data();
    // Claim owned recipe components depth-first in pre-order (an owned component covers its subtree).
    bool taken[N_SLOTS] = {}, covered[MAX_RECIPE_NODES] = {};
    int32_t discount = 0;
    for (int k = 0; k < MAX_RECIPE_NODES; ++k) {
        int it = (int)cat.node_item[(size_t)row * MAX_RECIPE_NODES + k];
        int parent = (int)cat.node_parent[(size_t)row * MAX_RECIPE_NODES + k];
        bool parent_cov = parent >= 0 ? covered[std::max(parent, 0)] : false;
        int slot = 0;
        bool any = false;
        for (int s = 0; s < N_SLOTS; ++s) {
            bool avail = items[s] == it && it >= 0 && !taken[s] && s < TRINKET_SLOT && stacks[s] > 0;
            if (avail && !any) slot = s;
            any = any || avail;
        }
        bool got = it >= 0 && !parent_cov && any;
        taken[slot] = taken[slot] || got;
        covered[k] = parent_cov || got;
        discount += got ? (int32_t)cat.node_total[(size_t)row * MAX_RECIPE_NODES + k] : 0;
    }
    float cost = (float)((int32_t)cat.total[row] - discount);
    int32_t rem_items[N_SLOTS], rem_stacks[N_SLOTS];
    for (int s = 0; s < N_SLOTS; ++s) rem_items[s] = taken[s] ? EMPTY : items[s], rem_stacks[s] = taken[s] ? 0 : stacks[s];

    bool trinket = cat.trinket[row] != 0.f;
    int max_stack = (int)cat.max_stack[row];
    bool has_stack = false, has_free = false;
    int stack_slot = 0, free_slot = 0;
    for (int s = 0; s < N_SLOTS; ++s) {
        bool main = s < TRINKET_SLOT;
        bool stackable = rem_items[s] == row && rem_stacks[s] < max_stack && main && max_stack > 1;
        bool free = rem_items[s] < 0 && main;
        if (stackable && !has_stack) stack_slot = s;
        if (free && !has_free) free_slot = s;
        has_stack = has_stack || stackable, has_free = has_free || free;
    }
    // Group limits count occupied slots; a trinket purchase replaces the trinket slot instead of adding.
    bool group_ok = true, cd_ok = true;
    int adds = has_stack ? 0 : 1;
    for (int g = 0; g < ng; ++g) {
        if (!cat.group(row, g)) continue;
        int held = 0;
        for (int s = 0; s < N_SLOTS; ++s) {
            bool present = rem_items[s] >= 0 && !(trinket && s == TRINKET_SLOT);
            held += present && cat.group(std::max(rem_items[s], 0), g);
        }
        int gmax = (int)cat.group_max[g];
        group_ok = group_ok && (gmax < 0 || held + adds <= gmax);
        cd_ok = cd_ok && now >= group_cd_until[g];
    }
    int slot = trinket ? TRINKET_SLOT : (has_stack ? stack_slot : free_slot);
    bool slot_ok = trinket || has_stack || has_free;
    int need_buff = (int)cat.required_buff[row];
    bool buff_ok = need_buff == -1 ||
                   (need_buff >= 0 && buff_currency.size() && buff_currency[std::max(need_buff, 0)]);
    bool purchasable = cat.in_store[row] != 0.f && cat.blocked[row] == 0.f && buff_ok &&
                       (cat.ranged_only[row] == 0.f || is_ranged);
    bool level_ok = (float)level >= cat.required_level[row];
    bool gold_ok = gold >= cost;
    int code = !can_shop ? ERR_NOT_IN_SHOP : !purchasable ? ERR_NOT_PURCHASABLE : !level_ok ? ERR_LEVEL
               : !cd_ok ? ERR_COOLDOWN : !group_ok ? ERR_GROUP : !slot_ok ? ERR_NO_SLOT : !gold_ok ? ERR_GOLD : OK;
    bool ok = code == OK;
    ShopResult r;
    r.inv = inv;
    if (ok) {
        for (int s = 0; s < N_SLOTS; ++s) r.inv.item[s] = rem_items[s], r.inv.stack[s] = rem_stacks[s];
        r.inv.item[slot] = row;
        r.inv.stack[slot] = has_stack && !trinket ? rem_stacks[slot] + 1 : 1;
    }
    r.gold = ok ? gold - cost : gold;
    r.group_cd_until = group_cd_until;
    if (ok)
        for (int g = 0; g < ng; ++g)
            if (cat.group(row, g) && cat.group_purchase_cd[g] > 0.f) r.group_cd_until[g] = now + cat.group_purchase_cd[g];
    r.ok = ok, r.code = code, r.spent = ok ? cost : 0.f;
    return r;
}

// inventory.sell: one unit from ``slot`` at its sell value
ShopResult sell(const Inventory& inv, float gold, int slot, bool can_shop) {
    const Cat& cat = K();
    int row = inv.item[slot];
    bool occupied = row >= 0;
    bool sellable = occupied && cat.can_be_sold[clip_row(std::max(row, 0))] != 0.f;
    int code = !can_shop ? ERR_NOT_IN_SHOP : !occupied ? ERR_EMPTY_SLOT : !sellable ? ERR_NOT_SELLABLE : OK;
    bool ok = code == OK;
    float value = ok ? (float)(int32_t)cat.sell_value[clip_row(std::max(row, 0))] : 0.f;
    int left = inv.stack[slot] - 1;
    ShopResult r;
    r.inv = inv;
    if (ok) r.inv.item[slot] = left > 0 ? row : EMPTY, r.inv.stack[slot] = std::max(left, 0);
    r.gold = gold + value, r.ok = ok, r.code = code, r.spent = -value;
    return r;
}

// inventory.replace_item: transform the first ``from_row`` in place
Inventory replace_item(const Inventory& inv, int from_row, int to_row, bool enabled) {
    Inventory out = inv;
    for (int s = 0; s < N_SLOTS; ++s)
        if (inv.item[s] == from_row && enabled) {
            out.item[s] = to_row;
            break;
        }
    return out;
}

// inventory.consume_one: use one unit of a consumed item
Inventory consume_one(const Inventory& inv, int slot, bool enabled) {
    Inventory out = inv;
    bool d = enabled && inv.item[slot] >= 0 && inv.stack[slot] > 0;
    int left = inv.stack[slot] - 1;
    out.item[slot] = d && left <= 0 ? EMPTY : inv.item[slot];
    out.stack[slot] = d ? std::max(left, 0) : inv.stack[slot];
    return out;
}

// ---- direct-check registrations (ops/native/test_hooks.py test_econ) ---------------------------------------------
namespace {
struct I1 {
    Arr<int32_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
struct B1 {
    Arr<uint8_t> v;
    template <class F> void visit(F&& f) { f(v); }
};
ShopResult buy_test(Inventory inv, float gold, int32_t row, uint8_t can_shop, int32_t level, uint8_t is_ranged, float now,
                    Arr<float> group_cd_until, Arr<uint8_t> buff_currency) {
    return buy(inv, gold, row, can_shop, level, is_ranged, now, group_cd_until, buff_currency);
}
ShopResult sell_test(Inventory inv, float gold, int32_t slot, uint8_t can_shop) { return sell(inv, gold, slot, can_shop); }
Inventory replace_item_test(Inventory inv, int32_t from_row, int32_t to_row, uint8_t enabled) {
    return replace_item(inv, from_row, to_row, enabled);
}
Inventory consume_one_test(Inventory inv, int32_t slot, uint8_t enabled) { return consume_one(inv, slot, enabled); }
I1 owned_counts_test(Inventory inv) { return I1{owned_counts(inv)}; }
ItemStats inventory_stats_test(Inventory inv) { return inventory_stats(inv); }
B1 in_shop_area_test(Arr<float> x, Arr<float> z, Arr<int32_t> team, Arr<uint8_t> dead) {
    B1 o{Arr<uint8_t>(x.size())};
    for (size_t i = 0; i < x.size(); ++i) o.v[i] = in_shop_area(x[i], z[i], team[i], dead[i]);
    return o;
}
Inventory inventory_from_ids_test(Arr<int32_t> ids, int32_t k, uint8_t trinket) { return inventory_from_ids(ids, k, trinket); }
}  // namespace
LANESIM_TEST(inventory_buy, "inventory.buy", buy_test);
LANESIM_TEST(inventory_sell, "inventory.sell", sell_test);
LANESIM_TEST(inventory_replace_item, "inventory.replace_item", replace_item_test);
LANESIM_TEST(inventory_consume_one, "inventory.consume_one", consume_one_test);
LANESIM_TEST(inventory_owned_counts, "inventory.owned_counts", owned_counts_test);
LANESIM_TEST(inventory_inventory_stats, "inventory.inventory_stats", inventory_stats_test);
LANESIM_TEST(inventory_in_shop_area, "inventory.in_shop_area", in_shop_area_test);
LANESIM_TEST(inventory_inventory_from_ids, "inventory.inventory_from_ids", inventory_from_ids_test);

}  // namespace lanesim::econ
