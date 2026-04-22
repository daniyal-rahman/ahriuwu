
UnknownCheats - Leading the game hacking and cheat development scene since 2000 
UnKnoWnCheaTs Game Hacking Portal UnKnoWnCheaTs Game Hacking Forum – Cheats, Hacks, and Tutorials Download Game Hacks, Cheats and Hacking Tools – UnKnoWnCheaTs Game Hacking Wiki – Tutorials and Guides on UnKnoWnCheaTs Toggle Dark Mode Register at UnKnoWnCheaTs – Join the Greatest Game Hacking Community

AD

Go Back   UnKnoWnCheaTs - Multiplayer Game Hacking and Cheats
MMO and Strategy Games
League of Legends
Reload this Page [Coding] League of Legends Reversal, Structs and Offsets
User Name:
Password:
Remember Me? 

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 649 of 651 « First < 149 549 599 639 645 646 647 648 649 650 651 > 

Thread Tools
Old 5th March 2026, 07:50 PM   #12961
docfpentakilk
n00bie

docfpentakilk's Avatar

Join Date: Sep 2021
Posts: 17
Reputation: 10
Rep Power: 112
docfpentakilk has made posts that are generally average in quality
Points: 3,674, Level: 6
Points: 3,674, Level: 6 Points: 3,674, Level: 6 Points: 3,674, Level: 6
Level up: 9%, 826 Points needed
Level up: 9% Level up: 9% Level up: 9%
Activity: 5.0%
Activity: 5.0% Activity: 5.0% Activity: 5.0%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Can someone please check which offset is incorrect in the latest version of the game?

inline std::uint64_t CharacterBaseData = 0x4030;
inline std::uint64_t CharData = 0x30;
inline std::uint64_t CharDataDataName = 0x40;
inline std::uint64_t CharDataDataSize = 0xB8;
inline std::uint64_t CharDataDataObjType = 0x7A0;
inline std::uint64_t CharDataDataObjTypeDetailed = 0x20;
inline std::uint64_t CharHealthbarheight = 0xB8;
inline std::uint64_t AtkSpeed ​​= 0x228;
docfpentakilk is offline

Old 6th March 2026, 01:12 AM   #12962
blackkx
God-Like

blackkx's Avatar

Join Date: Dec 2022
Posts: 154
Reputation: -40
Rep Power: 0
blackkx is becoming a waste of our time
Points: 3,975, Level: 6
Points: 3,975, Level: 6 Points: 3,975, Level: 6 Points: 3,975, Level: 6
Level up: 42%, 525 Points needed
Level up: 42% Level up: 42% Level up: 42%
Activity: 7.5%
Activity: 7.5% Activity: 7.5% Activity: 7.5%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
anyone know to get Cosmic Insight?
How do I calculate Flash cooldown with Cosmic Insight?
Last edited by Altoid; 6th March 2026 at 03:56 AM. Reason: Restored by Altoid
blackkx is online now

Old 6th March 2026, 08:37 AM   #12963
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by Raxrot View Post
hi dear, how are you reading the MISSILE OBJECT offsets you shared? i am currently reading StartPos correctly using missileObj + 0x388.
Why are your offsets different? Is there a pointer chain starting from 0x1C0?
My code:
missile.startPos = process.read<Vector3>(missileObjAddr + 0x388);

u can recheck verify it, my dump but not check it

but I recheck verify

Code:
namespace Missile {
    constexpr auto SpellCastPtr     = 0x8;          // [S] ptr to SpellCast object
    constexpr auto CastInfoBase     = 0x318;        // [IDA] sub_24E4E0: lea rcx,[r15+318h] (was 0x2C0→0x1C0 WRONG)
    constexpr auto SpellDataInst    = 0x318;        // [IDA] first QWORD at CastInfoBase = SpellData ptr
    constexpr auto SpellName        = 0x338;        // [IDA] CastInfo+0x20 (obfuscated ptr/value at obj+0x338)
    constexpr auto MissileName      = 0x360;        // [IDA] CastInfo+0x48 (string init call sub_322440 confirmed)
    constexpr auto StartPos         = 0x388;        // [IDA] CastInfo+0x70 vec3 (confirmed by Raxrot + IDA float access)
    constexpr auto EndPos           = 0x394;        // [IDA] CastInfo+0x7C vec3 (confirmed: movss xmm0,[r10+394h])
    constexpr auto CastEndPos       = 0x3A4;        // [IDA] CastInfo+0x8C vec3
    constexpr auto CasterNetId      = 0x3B0;        // [IDA] CastInfo+0x98 int (source caster network id)
    constexpr auto NetworkId        = 0x3BC;        // [IDA] CastInfo+0xA4 int (missile network id)
    constexpr auto Position         = 0x25C;        // [S] inherited vec3 position (GameObject base)
}
Last edited by trankhanhtinh1; 6th March 2026 at 10:06 AM.
trankhanhtinh1 is offline

Old 7th March 2026, 10:34 AM   #12964
tsanummy
n00bie

tsanummy's Avatar

Join Date: Mar 2026
Posts: 6
Reputation: 10
Rep Power: 3
tsanummy has made posts that are generally average in quality
Points: 29, Level: 1
Points: 29, Level: 1 Points: 29, Level: 1 Points: 29, Level: 1
Level up: 8%, 371 Points needed
Level up: 8% Level up: 8% Level up: 8%
Activity: 17.5%
Activity: 17.5% Activity: 17.5% Activity: 17.5%
there has been numerous leaked pdb and macos debug symbols, also some debug builds even in the past, does anyone keep a stash of them and is willing to share?
tsanummy is offline

Old 7th March 2026, 11:39 AM   #12965
BBasset
Super H4x0r

BBasset's Avatar

Join Date: Aug 2019
Posts: 333
Reputation: 2023
Rep Power: 168
BBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating community
Points: 10,517, Level: 12
Points: 10,517, Level: 12 Points: 10,517, Level: 12 Points: 10,517, Level: 12
Level up: 60%, 483 Points needed
Level up: 60% Level up: 60% Level up: 60%
Activity: 5.3%
Activity: 5.3% Activity: 5.3% Activity: 5.3%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
u can recheck verify it, my dump but not check it

but I recheck verify

Code:
namespace Missile {
    constexpr auto SpellCastPtr     = 0x8;          // [S] ptr to SpellCast object
    constexpr auto CastInfoBase     = 0x318;        // [IDA] sub_24E4E0: lea rcx,[r15+318h] (was 0x2C0→0x1C0 WRONG)
    constexpr auto SpellDataInst    = 0x318;        // [IDA] first QWORD at CastInfoBase = SpellData ptr
    constexpr auto SpellName        = 0x338;        // [IDA] CastInfo+0x20 (obfuscated ptr/value at obj+0x338)
    constexpr auto MissileName      = 0x360;        // [IDA] CastInfo+0x48 (string init call sub_322440 confirmed)
    constexpr auto StartPos         = 0x388;        // [IDA] CastInfo+0x70 vec3 (confirmed by Raxrot + IDA float access)
    constexpr auto EndPos           = 0x394;        // [IDA] CastInfo+0x7C vec3 (confirmed: movss xmm0,[r10+394h])
    constexpr auto CastEndPos       = 0x3A4;        // [IDA] CastInfo+0x8C vec3
    constexpr auto CasterNetId      = 0x3B0;        // [IDA] CastInfo+0x98 int (source caster network id)
    constexpr auto NetworkId        = 0x3BC;        // [IDA] CastInfo+0xA4 int (missile network id)
    constexpr auto Position         = 0x25C;        // [S] inherited vec3 position (GameObject base)
}
does it matches? doesn't look right except position

Code:
    namespace Missile {
        constexpr ULONGLONG Position = 0x25C;            // [confirmed] GameObject base vec3
        constexpr ULONGLONG SpellInfo = 0x2C0;           // [brute confirmed] ptr → +0x28 → char*spellName
        constexpr ULONGLONG SpellInfoNamePtr = 0x28;     // [brute confirmed] SpellInfo+0x28 → char*
        constexpr ULONGLONG StartPos = 0x388;            // [confirmed] CastInfo vec3
        constexpr ULONGLONG EndPos = 0x394;              // [confirmed] CastInfo vec3
        constexpr ULONGLONG CastEndPos = 0x3A4;          // [confirmed] CastInfo vec3
        // CasterNetId where??????
        // DestNetId where?????
    }
BBasset is offline

AD
Old 7th March 2026, 09:50 PM   #12966
Alexis913
n00bie

Alexis913's Avatar

Join Date: Nov 2024
Posts: 17
Reputation: 10
Rep Power: 35
Alexis913 has made posts that are generally average in quality
Points: 1,034, Level: 2
Points: 1,034, Level: 2 Points: 1,034, Level: 2 Points: 1,034, Level: 2
Level up: 27%, 366 Points needed
Level up: 27% Level up: 27% Level up: 27%
Activity: 12.5%
Activity: 12.5% Activity: 12.5% Activity: 12.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Hello everyone,

I'm back after a few months without playing.
I've updated the offsets (many patterns were still valid), but I see that I can't read the ViewMatrix values (0x1E2C030) because it returns an error. I don't have any issues with the other offsets.

Has there been any change to this?

Code:
inline constexpr std::intptr_t pMatrixBase                      = 0x1E2C030;
inline constexpr std::intptr_t oViewMatrix                      = 0x1AC;
inline constexpr std::intptr_t oProjectionMatrix                = 0x22C;
Code:
    std::vector<std::vector<float>> getViewProjMatrixFromGame(MemoryManager& mem, uintptr_t baseAddress)
    {
        auto matrixBaseOpt = mem.read<uintptr_t>(baseAddress + Offsets::pMatrixBase);
        if (!matrixBaseOpt || *matrixBaseOpt == 0) return {};

        const uintptr_t matrixBaseAddress = *matrixBaseOpt;
 
        std::vector<float> viewMatrix(16), projectionMatrix(16);
 
        for (int i = 0; i < 16; ++i) {
            auto valOpt = mem.read<float>(matrixBaseAddress + Offsets::oViewMatrix + i * sizeof(float));
            if (!valOpt) {
                return {}; <--- ERROR
            }
            viewMatrix[i] = *valOpt;
        }
 
        for (int i = 0; i < 16; ++i) {
            auto valOpt = mem.read<float>(matrixBaseAddress + Offsets::oProjectionMatrix + i * sizeof(float));
            if (!valOpt) {
                return {};
            }
            projectionMatrix[i] = *valOpt;
        }
 
        std::vector<std::vector<float>> viewMatrix2D(4, std::vector<float>(4));
        std::vector<std::vector<float>> projectionMatrix2D(4, std::vector<float>(4));
 
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                viewMatrix2D[i][j] = viewMatrix[i * 4 + j];
                projectionMatrix2D[i][j] = projectionMatrix[i * 4 + j];
            }
 
        std::vector<std::vector<float>> viewProjMatrix(4, std::vector<float>(4, 0.0f));
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                for (int k = 0; k < 4; ++k)
                    viewProjMatrix[row][col] += viewMatrix2D[row][k] * projectionMatrix2D[k][col];
 
        return viewProjMatrix;
    }
Last edited by Alexis913; 7th March 2026 at 09:52 PM.
Alexis913 is offline

Old 8th March 2026, 09:00 PM   #12967
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by BBasset View Post
does it matches? doesn't look right except position

Code:
    namespace Missile {
        constexpr ULONGLONG Position = 0x25C;            // [confirmed] GameObject base vec3
        constexpr ULONGLONG SpellInfo = 0x2C0;           // [brute confirmed] ptr → +0x28 → char*spellName
        constexpr ULONGLONG SpellInfoNamePtr = 0x28;     // [brute confirmed] SpellInfo+0x28 → char*
        constexpr ULONGLONG StartPos = 0x388;            // [confirmed] CastInfo vec3
        constexpr ULONGLONG EndPos = 0x394;              // [confirmed] CastInfo vec3
        constexpr ULONGLONG CastEndPos = 0x3A4;          // [confirmed] CastInfo vec3
        // CasterNetId where??????
        // DestNetId where?????
    }
yes, it like this

Code:
    constexpr auto SpellDataPtr     = 0x128;        // [IDA] sub_49E9F0: *(missile+0x128) = SpellData ptr
    constexpr auto Position         = 0x25C;        // [IDA] sub_90A0E0: Vec3 pos (inherited from GameObject)
    constexpr auto CastInfoBase     = 0x2C0;        // [IDA] sub_886AE0: CastInfo struct INLINE here (NOT a pointer!)
    constexpr auto MissileNetId     = 0x364;        // [IDA] sub_886AE0: [rsi+364h] = NetID (tree key) = CI+0xA4

    // --- CastInfo fields — ABSOLUTE offsets from missile base (0x2C0 + CI_*) ---
    //   Read directly: value = Read<T>(missile + offset)
    constexpr auto CI_SpellData     = 0x2C0;        // [IDA] QWORD: SpellData ptr (CastInfo+0x00)
    constexpr auto SpellName        = 0x2E0;        // [IDA] std::string SSO: spell name (CastInfo+0x20)
    constexpr auto MissileName      = 0x308;        // [IDA] std::string SSO: missile name (CastInfo+0x48)
    constexpr auto StartPos         = 0x330;        // [IDA] Vec3: start position (CastInfo+0x70)
    constexpr auto EndPos           = 0x33C;        // [IDA] Vec3: end position (CastInfo+0x7C)
    constexpr auto CastEndPos       = 0x34C;        // [IDA] Vec3: cast end position (CastInfo+0x8C)
    constexpr auto CasterNetId      = 0x358;        // [IDA] int: source caster net id (CastInfo+0x98)
    constexpr auto TargetNetId      = 0x35C;        // [IDA] int: target net id (CastInfo+0x9C)
    constexpr auto CI_TargetNetId2  = 0x360;        // [IDA] int: secondary target (CastInfo+0xA0)
    constexpr auto CI_MissileNetId  = 0x364;        // [IDA] int: missile net id (CastInfo+0xA4)
Quote:
Originally Posted by Alexis913 View Post
Hello everyone,

I'm back after a few months without playing.
I've updated the offsets (many patterns were still valid), but I see that I can't read the ViewMatrix values (0x1E2C030) because it returns an error. I don't have any issues with the other offsets.

Has there been any change to this?

Code:
inline constexpr std::intptr_t pMatrixBase                      = 0x1E2C030;
inline constexpr std::intptr_t oViewMatrix                      = 0x1AC;
inline constexpr std::intptr_t oProjectionMatrix                = 0x22C;
Code:
    std::vector<std::vector<float>> getViewProjMatrixFromGame(MemoryManager& mem, uintptr_t baseAddress)
    {
        auto matrixBaseOpt = mem.read<uintptr_t>(baseAddress + Offsets::pMatrixBase);
        if (!matrixBaseOpt || *matrixBaseOpt == 0) return {};

        const uintptr_t matrixBaseAddress = *matrixBaseOpt;
 
        std::vector<float> viewMatrix(16), projectionMatrix(16);
 
        for (int i = 0; i < 16; ++i) {
            auto valOpt = mem.read<float>(matrixBaseAddress + Offsets::oViewMatrix + i * sizeof(float));
            if (!valOpt) {
                return {}; <--- ERROR
            }
            viewMatrix[i] = *valOpt;
        }
 
        for (int i = 0; i < 16; ++i) {
            auto valOpt = mem.read<float>(matrixBaseAddress + Offsets::oProjectionMatrix + i * sizeof(float));
            if (!valOpt) {
                return {};
            }
            projectionMatrix[i] = *valOpt;
        }
 
        std::vector<std::vector<float>> viewMatrix2D(4, std::vector<float>(4));
        std::vector<std::vector<float>> projectionMatrix2D(4, std::vector<float>(4));
 
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                viewMatrix2D[i][j] = viewMatrix[i * 4 + j];
                projectionMatrix2D[i][j] = projectionMatrix[i * 4 + j];
            }
 
        std::vector<std::vector<float>> viewProjMatrix(4, std::vector<float>(4, 0.0f));
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                for (int k = 0; k < 4; ++k)
                    viewProjMatrix[row][col] += viewMatrix2D[row][k] * projectionMatrix2D[k][col];
 
        return viewProjMatrix;
    }
why u don't try use my dll inject for dump
trankhanhtinh1 is offline

Old 8th March 2026, 10:18 PM   #12968
Alexis913
n00bie

Alexis913's Avatar

Join Date: Nov 2024
Posts: 17
Reputation: 10
Rep Power: 35
Alexis913 has made posts that are generally average in quality
Points: 1,034, Level: 2
Points: 1,034, Level: 2 Points: 1,034, Level: 2 Points: 1,034, Level: 2
Level up: 27%, 366 Points needed
Level up: 27% Level up: 27% Level up: 27%
Activity: 12.5%
Activity: 12.5% Activity: 12.5% Activity: 12.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
yes, it like this

why u don't try use my dll inject for dump
Because I didn't know that was public . I searched for (LOLDumper v3.0 + offsetplugin.hpp + IDA MCP) but I didn't find anything.

Anyway, the problem is not the offset value, since my scanner finds the same value for that offset (pMatrixBase = 0x1E2C030). The problem is that I can't read it from an external script, even though I can read all the other offsets (HP, attack speed, level, ...).

Alexis913 is offline

Old 10th March 2026, 05:48 AM   #12969
msfool
Senior Member

msfool's Avatar

Join Date: Jun 2023
Posts: 81
Reputation: 10
Rep Power: 68
msfool has made posts that are generally average in quality
Points: 2,236, Level: 4
Points: 2,236, Level: 4 Points: 2,236, Level: 4 Points: 2,236, Level: 4
Level up: 20%, 564 Points needed
Level up: 20% Level up: 20% Level up: 20%
Activity: 5.0%
Activity: 5.0% Activity: 5.0% Activity: 5.0%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by kyudev View Post
I dont think so, I saw a lot of python opencv orbwalkers that used pynput and they claim they didnt get banned. The Vanguard they are using here is nowhere near the Vanguard of Valorant.

do you know if the signatures in EUW and CH are different? I cant find a working pattern for herolist this is pain none of these work anymore:
Code:
48 ? ? ? ? ? ? 48 ? ? ? ? 33 c0 89 ? ? ? 89 ? ? ? e8 ? ? ? ? 8b
48 8B 0D ? ? ? ? 0F 85 ? ? ? ? 83
48 8B 05 ? ? ? ? 45 33 E4 49 89 5B 08
48 8B 0D ? ? ? ? E8 ? ? ? ? 48 85 C0 74 33 8B 95
{ &OFFSETS::HeroManager, SIGS_TYPE_BASE, 3, "48 8B 0D ? ? ? ? 4C 8D 44 24 ? 33 C0 48 8D 54 24"},
msfool is offline

Old 10th March 2026, 08:22 PM   #12970
Alexis913
n00bie

Alexis913's Avatar

Join Date: Nov 2024
Posts: 17
Reputation: 10
Rep Power: 35
Alexis913 has made posts that are generally average in quality
Points: 1,034, Level: 2
Points: 1,034, Level: 2 Points: 1,034, Level: 2 Points: 1,034, Level: 2
Level up: 27%, 366 Points needed
Level up: 27% Level up: 27% Level up: 27%
Activity: 12.5%
Activity: 12.5% Activity: 12.5% Activity: 12.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
{ &OFFSETS::HeroManager, SIGS_TYPE_BASE, 3, "48 8B 0D ? ? ? ? 4C 8D 44 24 ? 33 C0 48 8D 54 24"},
That doesn't work because it gives you 2 results.

I use this pattern:
48 8B 3D ?? ?? ?? ?? FF CA
Last edited by Alexis913; 10th March 2026 at 08:55 PM.
Alexis913 is offline

Old 14th March 2026, 06:21 PM   #12971
kral84
n00bie

kral84's Avatar

Join Date: Mar 2015
Posts: 9
Reputation: -120
Rep Power: 0
kral84 is an outcastkral84 is an outcast
Points: 7,965, Level: 10
Points: 7,965, Level: 10 Points: 7,965, Level: 10 Points: 7,965, Level: 10
Level up: 34%, 735 Points needed
Level up: 34% Level up: 34% Level up: 34%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Cool
Hello,

Quote:
// LOLDumper v3.0 - FULL GAME DUMP (with base-tracking fix)
// Generated: Sat Mar 14 21:13:16 2026
// Module base: 0x7ff692ca0000
// Module size: 0x202d000
// ================================================================

// ==================== GLOBAL POINTERS ====================
# define oLocalPlayer 0x1dab760
# define oHerroList 0x1d7a470
# define oGametime 0x1d88580
# define oMissileList 0x1d7dd90
# define oNavGrid 0x1d7dd08
# define oHudInstance 0x1d7a5b8
# define oUnderMouseObj 0x1d7df90
# define ViewPort 0x1d8d1f0
# define IssueOrderFlag 0x1cddf88
# define CastSpellFlag 0x1cddf20
// FAILED: oMinionManager
# define oObjectManager 0x1d7a418
# define oViewPort2 0x1e3feb8
# define oMySpellState 0x1d80ae0
// FAILED: oKeyBoardHit
# define oMouseScreenVec2 0x1d7dd38

// ==================== FUNCTIONS ====================
# define oIssueOrder 0x29fc20
# define AttackDelay 0x52c5a0
# define GetPing 0x669eb0
# define WorldToScreen 0x1241600
// FAILED: isMinion
# define oGetAttackWindup 0x52c4a0
# define oGetBoundingRadius 0x285650
# define isTurret 0x308600
# define GetFirstObject 0x512920
# define GetNextObject 0x513410
# define oCastSpellWrapper 0x1e9a80
# define oGetCollisionFlags 0x1195e10
# define oGetAiManager 0x50aae0
# define oPrintChat 0xafe2e0
# define IsAlive 0x2e6360
# define IsHero 0x308700
// FAILED: CastSpell2

// ==================== STRUCT OFFSETS (Pattern-based) ====================
// --- AiManager ---
# define oObjAiManager 0x4038
# define oAiManagerStartPath 0x88
# define oAiManagerEndPath 0x88
// --- BasicAttack ---
# define oBasicAttackBase 0x2c68
# define oBasicAttackOffset1 0x2c0
# define oBasicAttackOffset2 0x70
// --- BuffManager ---
# define oObjBuffManager 0x28b8
// --- GameObject ---
# define oObjNetId 0xcc
# define oObjPosition 0xcc
# define oObjRadius 0x6f8
# define TeamID 0x259
# define oTargetable 0xed0
# define NamePlayer 0x4330
// --- HUD ---
# define oHudSpell 0x68
# define oHudMouse 0x28
// --- Health_Verified ---
# define oHealth_base 0x1080
# define oHealth_mHP 0x1080
# define oMaxHealth_mMaxHP 0x10a8
# define oHPMaxPenalty 0x10d0
# define oAllShield 0x1120
# define oPhysicalShield 0x1148
# define oMagicalShield 0x1170
# define oChampSpecificHealth 0x1198
// --- Minion ---
# define LaneMinionArray 0x68
# define LaneMinionType 0x4c79
// --- Missile ---
# define oMissileCastInfo 0x1c0
// --- NavGrid ---
# define oNavGridMinX 0x30
// --- SpellBook ---
# define oObjSpellBook 0x30e8
# define oObjSpellBookSpellSlot 0xae0
// --- SpellData ---
# define oSpellDataResource 0x8
// --- SpellSlot ---
# define oSpellSlotSpellInfo 0x128
// --- StatBlock_Verified ---
# define oHeroStatBase 0x1b88
# define oArmor 0x2060
# define oSpellBlock 0x20b0
# define oBaseAttackDamage 0x1ed0
# define oAttackRange 0x21a0
# define oMoveSpeed 0x2150
# define oAttackSpeedMod 0x1e30
# define oCrit 0x2010
# define oHPRegenRate 0x2100
# define oPercentCooldownMod 0x1b88
# define oAbilityHasteMod 0x1bb0
# define oFlatMagicPenetration 0x2308
# define oPercentMagicPenetration 0x2358
# define oFlatArmorPenetration 0x2218
# define oPercentArmorPenetration 0x2268
# define oPercentLifeSteal 0x23a8
# define oBonusArmor 0x2088
# define oBonusSpellBlock 0x20d8
# define oFlatPhysicalDamageMod 0x1cc8
# define oFlatMagicDamageMod 0x1d68
# define oBaseAbilityDamage 0x1f70

// ================================================================
// REPLICATED PROPERTIES (146 total)
// v3: Offsets are ABSOLUTE from game object (base+delta resolved)
// ================================================================

// --- sub_1FC2F0 (11 properties) [stat base=0x1b88] ---
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_1FC4C0 (12 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mPrimaryARRegenRateRep 0x2510 // base 0x1b88 + 0x988

// --- sub_1FC6B0 (18 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mMoveSpeedBaseIncrease 0x2178 // base 0x1b88 + 0x5f0
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668

// --- sub_1FC9C0 (27 properties) [stat base=0x1b88] ---
# define o_mPassiveCooldownEndTime 0x1c00 // base 0x1b88 + 0x78
# define o_mPassiveCooldownTotalTime 0x1c28 // base 0x1b88 + 0xa0
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mFlatCastRangeMod 0x1e08 // base 0x1b88 + 0x280
# define o_mPercentCooldownMod 0x1e08 // base 0x1b88 + 0x280
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mBaseAbilityDamage 0x1f70 // base 0x1b88 + 0x3e8
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatArmorPenetration 0x2218 // base 0x1b88 + 0x690
# define o_mPercentArmorPenetration 0x2268 // base 0x1b88 + 0x6e0
# define o_mFlatMagicPenetration 0x2308 // base 0x1b88 + 0x780
# define o_mPercentMagicPenetration 0x2358 // base 0x1b88 + 0x7d0
# define o_mPercentLifeStealMod 0x23a8 // base 0x1b88 + 0x820
# define o_mPercentSpellVampMod 0x23d0 // base 0x1b88 + 0x848
# define o_mPercentPhysicalVamp 0x2420 // base 0x1b88 + 0x898
# define o_mPARRegenRate 0x2510 // base 0x1b88 + 0x988

// --- sub_1FCDC0 (30 properties) [stat base=0x1b88] ---
# define o_mAbilityHasteMod 0x1bb0 // base 0x1b88 + 0x28
# define o_mPercentCooldownCapMod 0x1bd8 // base 0x1b88 + 0x50
# define o_mPercentBonusPhysicalDamageMod 0x1d18 // base 0x1b88 + 0x190
# define o_mPercentBasePhysicalDamageAsFlatBonusMod 0x1d40 // base 0x1b88 + 0x1b8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mPercentHealingAmountMod 0x1ea8 // base 0x1b88 + 0x320
# define o_mBaseAttackDamageSansPercentScale 0x1ef8 // base 0x1b88 + 0x370
# define o_mFlatBaseAttackDamageMod 0x1f20 // base 0x1b88 + 0x398
# define o_mPercentBaseAttackDamageMod 0x1f48 // base 0x1b88 + 0x3c0
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mFlatBaseHPPoolMod 0x2038 // base 0x1b88 + 0x4b0
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mBaseHPRegenRate 0x2128 // base 0x1b88 + 0x5a0
# define o_mPhysicalLethality 0x2240 // base 0x1b88 + 0x6b8
# define o_mPercentBonusArmorPenetration 0x2290 // base 0x1b88 + 0x708
# define o_mPercentCritBonusArmorPenetration 0x22b8 // base 0x1b88 + 0x730
# define o_mPercentCritTotalArmorPenetration 0x22e0 // base 0x1b88 + 0x758
# define o_mMagicLethality 0x2330 // base 0x1b88 + 0x7a8
# define o_mPercentBonusMagicPenetration 0x2380 // base 0x1b88 + 0x7f8
# define o_mPercentOmnivampMod 0x23f8 // base 0x1b88 + 0x870
# define o_mPercentCCReduction 0x2470 // base 0x1b88 + 0x8e8
# define o_mPercentEXPBonus 0x2498 // base 0x1b88 + 0x910
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mPrimaryARBaseRegenRateRep 0x2538 // base 0x1b88 + 0x9b0
# define o_mSecondaryARRegenRateRep 0x2560 // base 0x1b88 + 0x9d8
# define o_mSecondaryARBaseRegenRateRep 0x2588 // base 0x1b88 + 0xa00
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_1FD270 (5 properties) [stat base=0x1b88] ---
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668
# define o_mPathfindingRadiusMod 0x2448 // base 0x1b88 + 0x8c0

// --- sub_2E3230 (10 properties) ---
# define o_mMaxHP 0x2800
# define o_mHPMaxPenalty 0x5000
# define o_mAllShield 0xa000
# define o_mPhysicalShield 0xc800
# define o_mMagicalShield 0xf000
# define o_mChampSpecificHealth 0x1180
# define o_mIncomingHealingAllied 0x1400
# define o_mIncomingHealingEnemy 0x1680
# define o_mIncomingDamage 0x1900
# define o_mHP 0x1080

// --- sub_2E38E0 (16 properties) ---
# define o_mMaxPAR 0x2800
# define o_mSAR 0x1080
# define o_mMaxSAR 0x1300
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPAR 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mGold 0x2830
# define o_mGoldTotal 0x2858
# define o_mMinimumGold 0x2880
# define o_mExp 0x4cf0
# define o_mVisionScore 0x55e0
# define o_mShutdownValue 0x5608
# define o_mBaseGoldGivenOnDeath 0x5630

// --- sub_2E4DD0 (15 properties) ---
// mMaxPAR (offset unknown)
// mPARState (offset unknown)
// mMaxPAR (offset unknown)
// mMaxSAR (offset unknown)
// mMaxSAR (offset unknown)
# define o_mPAR 0x1080
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mPercentDamageToBarracksMinionMod 0x1c50
# define o_mFlatDamageReductionFromBarracksMinionMod 0x1c78
# define o_mIncreasedMoveSpeedMinionMod 0x1ca0
# define o_mFollowerTargetDelay 0x2db8

// --- sub_35B070 (2 properties) ---
# define o_mMP 0x3600
# define o_mMaxMP 0x3880
// --- AiManager Internal (offsets from AiManager*) ---
// AiManager* = *(gameObj + oObjAiManager)
# define oAiManagerTargetPos 0x34 // vec3 (target position)
# define oAiManagerVelocity 0x318 // float (movement speed scalar)
# define oAiManagerIsMoving 0x31C // float/bool
# define oAiManagerCurrentSegment 0x320 // int32
# define oAiManagerStartPath 0x330 // vec3 (path start)
# define oAiManagerTargetPosition 0x33C // vec3 (= StartPath + 0xC)
# define oAiManagerNavArray 0x348 // ptr to vec3[] waypoints
# define oAiManagerSegmentsCount 0x350 // int32 (waypoint count)
# define oAiManagerDashSpeed 0x360 // float
# define oAiManagerIsDashing 0x384 // byte/bool
# define oAiManagerServerPos 0x474 // vec3 (server position)
# define oAiManagerMoveVec3 0x480 // vec3 (StartPath + 0x150)

// --- BuffManager Internal (offsets from BuffManager*) ---
// BuffManager* = gameObj + oObjBuffManager
# define oBuffManagerArray 0x18 // ptr to buff entry array start
# define oBuffManagerArrayEnd 0x20 // ptr to buff entry array end
// BuffInstance struct:
# define oBuffInstanceType 0x0C // int (buff type: 24/25/26 = valid)
# define oBuffInstanceScript 0x10 // ptr -> BuffScript
# define oBuffInstanceStartTime 0x18 // float (game time start)
# define oBuffInstanceEndTime 0x1C // float (game time end, 25000+ = perm)
# define oBuffInstanceDuration 0x20 // float (duration)
# define oBuffInstanceStackCount 0x38 // int (stack count)
# define oBuffInstanceCount 0x78 // int (instance count)
// BuffScript struct:
# define oBuffScriptName 0x08 // char* (buff name string)

// --- SpellSlot Internal (offsets from SpellSlot*) ---
// SpellSlot* = *(gameObj + oObjSpellBook + oObjSpellBookSpellSlot + 8*slotIdx)
# define oSpellSlotLevel 0x28 // int32 (spell level)
# define oSpellSlotReadyTime 0x30 // float (next ready game time)
# define oSpellSlotStartTime 0x34 // float (cast start game time)
# define oSpellSlotCoolTime 0x74 // float (last cooldown duration)
# define oSpellSlotSpellInfo 0x128 // ptr to SpellInfo
# define oSpellSlotSpellInput 0x120 // ptr to SpellInput

// --- SpellInfo/SpellData/SpellDataResource chain ---
# define oSpellInfoSpellData 0x00 // SpellInfo -> SpellData
# define oSpellInfoSrcIndex 0x88 // int (caster NetID index)
# define oSpellInfoStartPos 0xA4 // vec3
# define oSpellInfoEndPos 0xB0 // vec3 (StartPos + 0xC)
# define oSpellInfoCastPos 0xBC // vec3 (EndPos + 0xC)
# define oSpellInfoTargetIndex 0xE0 // int (target NetID index)
# define oSpellInfoCastDelay 0xF0 // float
# define oSpellInfoIsSpell 0x10C // byte (== 0 for spell)
# define oSpellInfoIsSpecialAttack 0x112 // byte
# define oSpellInfoIsAutoAttack 0x113 // byte
# define oSpellInfoSlot 0x11C // int (spell slot index)
# define oSpellDataScript 0x18 // SpellData -> SpellDataScript
# define oSpellDataScriptName 0x08 // SpellDataScript -> name string
# define oSpellDataSpellName 0x28 // SpellData -> spell name string
# define oSpellDataResource 0x60 // SpellData -> SpellDataResource

// --- SpellInput (from SpellSlot + oSpellSlotSpellInput) ---
# define oSpellInputMeVec 0x18 // vec3 (caster position)
# define oSpellInputStartVec1 0x24 // vec3 (MeVec + 0xC)
# define oSpellInputStartVec2 0x30 // vec3 (StartVec1 + 0xC)
# define oSpellInputEndVec 0x3C // vec3 (StartVec2 + 0xC)
# define oSpellInputTargetVec 0x24 // vec3

// --- MissileData (offsets from Missile game object) ---
# define oMissileSpellInfo 0x260 // ptr to SpellInfo
# define oMissileSpellName 0x2E0 // std::string (spell name)
# define oMissileMissileName 0x308 // std::string (missile name)
# define oMissileStartPos 0x2E0 // vec3
# define oMissileEndPos 0x2EC // vec3
# define oMissileSrcIdx 0x2C4 // int (caster NetID)
# define oMissileDestIdx 0x318 // int (target NetID)
# define oMissilePosition 0x25C // vec3 (current, inherited from GameObject)

// --- OnCastingSpell (from SpellBook + 0x38) ---
# define oOnCastSpellInfo 0x08 // ptr to SpellInfo
# define oOnCastSpellName 0x28 // spell name string
# define oOnCastStartPosition 0xD0 // vec3
# define oOnCastTargetPosition 0xDC // vec3 (StartPos + 0xC)

// --- EffectEmitter ---
# define oEffectEmitterData 0x260
# define oEffectEmitterName 0x60
# define oEffectEmitterCaster 0x40
# define oEffectEmitterFirstCaster 0x08
# define oEffectEmitterTarget 0x30
# define oEffectEmitterFirstTarget 0x08

// --- HUD/Camera (offsets from HudInstance*) ---
# define oHudInstanceCamera 0x18
# define oHudInstanceInput 0x28
# define oHudInstanceUserData 0x60

// --- Zoom (from partern reference) ---
# define oZoom 0x28 // float (inside zoom struct)
# define oZoomCoefficient_o1 0x18 // camera chain offset 1
# define oZoomCoefficient_o2 0x318 // camera chain offset 2

// --- Object Manager Internal ---
# define oMgrSizeVec 0x10 // vector array size
# define oMgrSizeList 0x20 // list array size
# define oMgrObj 0x08 // object pointer

// --- GameObject Misc (not pattern-scannable) ---
# define oObjLiveState 0x43 // byte (live state)
# define oObjEffectData 0x260 // effect emitter data
# define oObjIsVisible 0x300 // byte (visibility)
# define oObjState 0x5A8 // int (channeling state, 0x400)
# define oObjOnCastingSpell (oObjSpellBook + 0x38) // SpellBook + 0x38
kral84 is offline

Old 14th March 2026, 06:55 PM   #12972
chen399516
n00bie

chen399516's Avatar

Join Date: Jan 2026
Posts: 17
Reputation: 10
Rep Power: 7
chen399516 has made posts that are generally average in quality
Points: 268, Level: 1
Points: 268, Level: 1 Points: 268, Level: 1 Points: 268, Level: 1
Level up: 67%, 132 Points needed
Level up: 67% Level up: 67% Level up: 67%
Activity: 9.4%
Activity: 9.4% Activity: 9.4% Activity: 9.4%
Does it have a TFT mode?
Quote:
Originally Posted by kral84 View Post
Hello,

// --- AiManager Internal (offsets from AiManager*) ---
// AiManager* = *(gameObj + oObjAiManager)
# define oAiManagerTargetPos 0x34 // vec3 (target position)
# define oAiManagerVelocity 0x318 // float (movement speed scalar)
# define oAiManagerIsMoving 0x31C // float/bool
# define oAiManagerCurrentSegment 0x320 // int32
# define oAiManagerStartPath 0x330 // vec3 (path start)
# define oAiManagerTargetPosition 0x33C // vec3 (= StartPath + 0xC)
# define oAiManagerNavArray 0x348 // ptr to vec3[] waypoints
# define oAiManagerSegmentsCount 0x350 // int32 (waypoint count)
# define oAiManagerDashSpeed 0x360 // float
# define oAiManagerIsDashing 0x384 // byte/bool
# define oAiManagerServerPos 0x474 // vec3 (server position)
# define oAiManagerMoveVec3 0x480 // vec3 (StartPath + 0x150)

// --- BuffManager Internal (offsets from BuffManager*) ---
// BuffManager* = gameObj + oObjBuffManager
# define oBuffManagerArray 0x18 // ptr to buff entry array start
# define oBuffManagerArrayEnd 0x20 // ptr to buff entry array end
// BuffInstance struct:
# define oBuffInstanceType 0x0C // int (buff type: 24/25/26 = valid)
# define oBuffInstanceScript 0x10 // ptr -> BuffScript
# define oBuffInstanceStartTime 0x18 // float (game time start)
# define oBuffInstanceEndTime 0x1C // float (game time end, 25000+ = perm)
# define oBuffInstanceDuration 0x20 // float (duration)
# define oBuffInstanceStackCount 0x38 // int (stack count)
# define oBuffInstanceCount 0x78 // int (instance count)
// BuffScript struct:
# define oBuffScriptName 0x08 // char* (buff name string)

// --- SpellSlot Internal (offsets from SpellSlot*) ---
// SpellSlot* = *(gameObj + oObjSpellBook + oObjSpellBookSpellSlot + 8*slotIdx)
# define oSpellSlotLevel 0x28 // int32 (spell level)
# define oSpellSlotReadyTime 0x30 // float (next ready game time)
# define oSpellSlotStartTime 0x34 // float (cast start game time)
# define oSpellSlotCoolTime 0x74 // float (last cooldown duration)
# define oSpellSlotSpellInfo 0x128 // ptr to SpellInfo
# define oSpellSlotSpellInput 0x120 // ptr to SpellInput

// --- SpellInfo/SpellData/SpellDataResource chain ---
# define oSpellInfoSpellData 0x00 // SpellInfo -> SpellData
# define oSpellInfoSrcIndex 0x88 // int (caster NetID index)
# define oSpellInfoStartPos 0xA4 // vec3
# define oSpellInfoEndPos 0xB0 // vec3 (StartPos + 0xC)
# define oSpellInfoCastPos 0xBC // vec3 (EndPos + 0xC)
# define oSpellInfoTargetIndex 0xE0 // int (target NetID index)
# define oSpellInfoCastDelay 0xF0 // float
# define oSpellInfoIsSpell 0x10C // byte (== 0 for spell)
# define oSpellInfoIsSpecialAttack 0x112 // byte
# define oSpellInfoIsAutoAttack 0x113 // byte
# define oSpellInfoSlot 0x11C // int (spell slot index)
# define oSpellDataScript 0x18 // SpellData -> SpellDataScript
# define oSpellDataScriptName 0x08 // SpellDataScript -> name string
# define oSpellDataSpellName 0x28 // SpellData -> spell name string
# define oSpellDataResource 0x60 // SpellData -> SpellDataResource

// --- SpellInput (from SpellSlot + oSpellSlotSpellInput) ---
# define oSpellInputMeVec 0x18 // vec3 (caster position)
# define oSpellInputStartVec1 0x24 // vec3 (MeVec + 0xC)
# define oSpellInputStartVec2 0x30 // vec3 (StartVec1 + 0xC)
# define oSpellInputEndVec 0x3C // vec3 (StartVec2 + 0xC)
# define oSpellInputTargetVec 0x24 // vec3

// --- MissileData (offsets from Missile game object) ---
# define oMissileSpellInfo 0x260 // ptr to SpellInfo
# define oMissileSpellName 0x2E0 // std::string (spell name)
# define oMissileMissileName 0x308 // std::string (missile name)
# define oMissileStartPos 0x2E0 // vec3
# define oMissileEndPos 0x2EC // vec3
# define oMissileSrcIdx 0x2C4 // int (caster NetID)
# define oMissileDestIdx 0x318 // int (target NetID)
# define oMissilePosition 0x25C // vec3 (current, inherited from GameObject)

// --- OnCastingSpell (from SpellBook + 0x38) ---
# define oOnCastSpellInfo 0x08 // ptr to SpellInfo
# define oOnCastSpellName 0x28 // spell name string
# define oOnCastStartPosition 0xD0 // vec3
# define oOnCastTargetPosition 0xDC // vec3 (StartPos + 0xC)

// --- EffectEmitter ---
# define oEffectEmitterData 0x260
# define oEffectEmitterName 0x60
# define oEffectEmitterCaster 0x40
# define oEffectEmitterFirstCaster 0x08
# define oEffectEmitterTarget 0x30
# define oEffectEmitterFirstTarget 0x08

// --- HUD/Camera (offsets from HudInstance*) ---
# define oHudInstanceCamera 0x18
# define oHudInstanceInput 0x28
# define oHudInstanceUserData 0x60

// --- Zoom (from partern reference) ---
# define oZoom 0x28 // float (inside zoom struct)
# define oZoomCoefficient_o1 0x18 // camera chain offset 1
# define oZoomCoefficient_o2 0x318 // camera chain offset 2

// --- Object Manager Internal ---
# define oMgrSizeVec 0x10 // vector array size
# define oMgrSizeList 0x20 // list array size
# define oMgrObj 0x08 // object pointer

// --- GameObject Misc (not pattern-scannable) ---
# define oObjLiveState 0x43 // byte (live state)
# define oObjEffectData 0x260 // effect emitter data
# define oObjIsVisible 0x300 // byte (visibility)
# define oObjState 0x5A8 // int (channeling state, 0x400)
# define oObjOnCastingSpell (oObjSpellBook + 0x38) // SpellBook + 0x38
Does it have a TFT mode?
chen399516 is offline

Old 14th March 2026, 06:58 PM   #12973
kral84
n00bie

kral84's Avatar

Join Date: Mar 2015
Posts: 9
Reputation: -120
Rep Power: 0
kral84 is an outcastkral84 is an outcast
Points: 7,965, Level: 10
Points: 7,965, Level: 10 Points: 7,965, Level: 10 Points: 7,965, Level: 10
Level up: 34%, 735 Points needed
Level up: 34% Level up: 34% Level up: 34%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
i dont know i used the LOLDUMPER this dll in this forum. but i think is wrong.
now i tested his

48 8B 3D ?? ?? ?? ?? FF CA

and ifound 0x2807DE632B0

DUMP: 0x2807DE632B0 (size: 0x80 / 128 bytes)

Offset | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | ASCII | Values
--------+-----------------------------------------------------+------------------+-------
+0000 | 20 A0 80 3A 80 02 00 00 30 B3 7C 2A 80 02 00 00 | ..:....0.|*.... | PTR 0x2803A80A020
+0010 | 38 DA 59 46 80 02 00 00 88 33 09 2A 80 02 00 00 | 8.YF.....3.*.... | PTR 0x2804659DA38
+0020 | 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 | ................ | NULL
+0030 | E0 36 7E 41 80 02 00 00 40 2C 09 2A 80 02 00 00 | .6~A....@,.*.... | PTR 0x280417E36E0
+0040 | 00 BD AA 7A 80 02 00 00 30 3A 3D 3E 80 02 00 00 | ...z....0:=>.... | PTR 0x2807AAABD00
+0050 | 18 31 16 8A F7 7F 00 00 58 EA F5 7D 80 02 00 00 | .1......X..}.... | i32: -1978257128, 32759
+0060 | 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 | ................ | NULL
+0070 | 18 31 16 8A F7 7F 00 00 70 EA F5 7D 80 02 00 00 | .1......p..}.... | i32: -1978257128, 32759

DUMP: 0x2803A80A020 (size: 0x100 / 256 bytes)

Offset | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | ASCII | Values
--------+-----------------------------------------------------+------------------+-------
+0000 | 28 21 0E 8A F7 7F 00 00 90 2C 0E 8A F7 7F 00 00 | (!.......,...... | i32: -1978785496, 32759
+0010 | A0 2C 0E 8A F7 7F 00 00 B8 2C 0E 8A F7 7F 00 00 | .,.......,...... | i32: -1978782560, 32759
+0020 | BC 02 01 00 00 00 00 00 CF 00 30 31 30 01 00 01 | ..........010... | PTR 0x102BC
+0030 | 03 0A 00 F5 F5 00 01 00 01 02 45 1E BA 1B 0E 01 | ..........E..... | i32: -184546813, 65781
+0040 | 00 01 01 7F 19 80 1E 40 01 00 01 01 BA 84 E1 F2 | ...... @........ | i32: 2130772224, 1075740697
+0050 | 45 6F 1A 0D 45 7B 1E 0D 45 7B 1A 0D 45 7F 1A 0D | Eo..E{..E{..E... | i32: 219836229, 220101445

---
220101342
+00C0 | 9F FF 60 00 00 01 00 01 01 00 00 00 16 00 00 40 | ..`............@ | i32: 6356895, 16777472
+00D0 | 42 61 73 41 67 72 C4 B1 73 C4 B1 00 00 00 00 00 | BasAgr..s....... | i32: 1098080578 (15.21f), -1312525721
+00E0 | 0B 00 00 00 00 00 00 00 0F 00 00 00 00 00 00 00 | ................ | i32: 11, 0
+00F0 | 00 7B 24 CA 00 00 00 00 1D C7 DF 45 1A 98 59 42 | .{$........E..YB | PTR 0xCA247B00
kral84 is offline

Old 14th March 2026, 07:10 PM   #12974
chen399516
n00bie

chen399516's Avatar

Join Date: Jan 2026
Posts: 17
Reputation: 10
Rep Power: 7
chen399516 has made posts that are generally average in quality
Points: 268, Level: 1
Points: 268, Level: 1 Points: 268, Level: 1 Points: 268, Level: 1
Level up: 67%, 132 Points needed
Level up: 67% Level up: 67% Level up: 67%
Activity: 9.4%
Activity: 9.4% Activity: 9.4% Activity: 9.4%
Quote:
Originally Posted by kral84 View Post
i dont know i used the LOLDUMPER this dll in this forum. but i think is wrong.
now i tested his

48 8B 3D ?? ?? ?? ?? FF CA

and ifound 0x2807DE632B0

DUMP: 0x2807DE632B0 (size: 0x80 / 128 bytes)

Offset | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | ASCII | Values
--------+-----------------------------------------------------+------------------+-------
+0000 | 20 A0 80 3A 80 02 00 00 30 B3 7C 2A 80 02 00 00 | ..:....0.|*.... | PTR 0x2803A80A020
+0010 | 38 DA 59 46 80 02 00 00 88 33 09 2A 80 02 00 00 | 8.YF.....3.*.... | PTR 0x2804659DA38
+0020 | 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 | ................ | NULL
+0030 | E0 36 7E 41 80 02 00 00 40 2C 09 2A 80 02 00 00 | .6~A....@,.*.... | PTR 0x280417E36E0
+0040 | 00 BD AA 7A 80 02 00 00 30 3A 3D 3E 80 02 00 00 | ...z....0:=>.... | PTR 0x2807AAABD00
+0050 | 18 31 16 8A F7 7F 00 00 58 EA F5 7D 80 02 00 00 | .1......X..}.... | i32: -1978257128, 32759
+0060 | 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 | ................ | NULL
+0070 | 18 31 16 8A F7 7F 00 00 70 EA F5 7D 80 02 00 00 | .1......p..}.... | i32: -1978257128, 32759

DUMP: 0x2803A80A020 (size: 0x100 / 256 bytes)

Offset | 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | ASCII | Values
--------+-----------------------------------------------------+------------------+-------
+0000 | 28 21 0E 8A F7 7F 00 00 90 2C 0E 8A F7 7F 00 00 | (!.......,...... | i32: -1978785496, 32759
+0010 | A0 2C 0E 8A F7 7F 00 00 B8 2C 0E 8A F7 7F 00 00 | .,.......,...... | i32: -1978782560, 32759
+0020 | BC 02 01 00 00 00 00 00 CF 00 30 31 30 01 00 01 | ..........010... | PTR 0x102BC
+0030 | 03 0A 00 F5 F5 00 01 00 01 02 45 1E BA 1B 0E 01 | ..........E..... | i32: -184546813, 65781
+0040 | 00 01 01 7F 19 80 1E 40 01 00 01 01 BA 84 E1 F2 | ...... @........ | i32: 2130772224, 1075740697
+0050 | 45 6F 1A 0D 45 7B 1E 0D 45 7B 1A 0D 45 7F 1A 0D | Eo..E{..E{..E... | i32: 219836229, 220101445

---
220101342
+00C0 | 9F FF 60 00 00 01 00 01 01 00 00 00 16 00 00 40 | ..`............@ | i32: 6356895, 16777472
+00D0 | 42 61 73 41 67 72 C4 B1 73 C4 B1 00 00 00 00 00 | BasAgr..s....... | i32: 1098080578 (15.21f), -1312525721
+00E0 | 0B 00 00 00 00 00 00 00 0F 00 00 00 00 00 00 00 | ................ | i32: 11, 0
+00F0 | 00 7B 24 CA 00 00 00 00 1D C7 DF 45 1A 98 59 42 | .{$........E..YB | PTR 0xCA247B00
Incorrect
chen399516 is offline

Old 14th March 2026, 07:18 PM   #12975
kral84
n00bie

kral84's Avatar

Join Date: Mar 2015
Posts: 9
Reputation: -120
Rep Power: 0
kral84 is an outcastkral84 is an outcast
Points: 7,965, Level: 10
Points: 7,965, Level: 10 Points: 7,965, Level: 10 Points: 7,965, Level: 10
Level up: 34%, 735 Points needed
Level up: 34% Level up: 34% Level up: 34%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
interesting working for me

Persistent RVA:0x1D7B048 → HeroManager pointer (pattern: 48 8B 3D ?? ?? ?? ?? FF CA)HeroManager struct:An array of 8-byte pointers starting at +0x00 → each one is a hero entity.Hero Entity offsets:OffsetFieldExample Value+0x00VTable0x7FF72BB52128+0x68Name (inline if ≤15, otherwise ptr)"Mordekaiser Bot"+0x80Name length15+0x88HP (float)190.0+0x8CMaxHP (float)125.0+0xF8Pos X (float)3385.04+0x100Pos Z (float)13039.5

RVA / OffsetFieldStatus0x1D7B048HeroManager (pattern: 48 8B 3D ?? ?? ?? ?? FF CA)
Entity +0x68Name
Entity +0x1080Current HP✓Entity +0xF8Position X
Entity +0x100Position Z
Entity +0x3BC0Start of spell slot pointer array
Spell +0x1CSpell Level
Spell +0x28Spell Slot Index
Spell +0x30Cooldown Expiry (GameTime)
Spell +0x74Base Cooldown
Spell +0x80Range
Spell +0x90Mana Cost
Last edited by kral84; 14th March 2026 at 08:26 PM.
kral84 is offline

Old 15th March 2026, 05:34 AM   #12976
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by chen399516 View Post
Incorrect
use version update for offset correct

The previous version was just a test, to find errors and make improvements. Now I can confirm that my LOLDUMPER dll can produce accurate results.

Code:
# pragma once
# include <cstdint>

// ================================================================
// League of Legends - Offsets
// Updated: 2026-03-12 (Hotfix) (LOLDumper v5.0 + offsetplugin.hpp + IDA MCP)
// Binary: League of Legends.exe
// Global/Function RVAs: from module size 0x202D000 (dump files)
// Struct offsets: verified via IDA on module size 0x2342000
// Base: 0x0 (relative offsets from module base)
//
// Sources:
//   [D]   = LOLDumper_full.h (pattern-scanned)
//   [P]   = offsetplugin.hpp (ida_lol_plugin.dll output)
//   [IDA] = IDA Pro MCP verified (decompile/disasm confirmed)
//   [CE]  = Cheat Engine verified at runtime
//   [S]   = struct offsets (unchanged between versions)
//   [C]   = chimera_structures.h reference (needs CE verify)
//
// Hotfix notes (2026-03-12):
//   - Function RVAs shifted +0x10 from 2026-03-05 hotfix
//   - All globals remained STABLE (confirmed via LOLDumper scan)
//   - Struct offsets STABLE (RegisterProperty-based, version-independent)
//   - LOLDumper re-scan confirmed globals unchanged
//   - Function deltas: IssueOrder +0x10, IsAlive +0x10, GetAttackDelay +0x10, etc.
//   - CastSpellSafe still at same RVA (offsetplugin.hpp: 0xBB9E60 → needs +0x10 verify)
//   - DetectionWatcher2 for Chimera-style mainloop_check is currently
//     resolved at runtime by signature: 4C 8B 3D ? ? ? ? 4D 85 FF 0F
//   - Current packet pipeline:
//       CastSpellSafe -> CastSpellPacketA/B/Charged -> PacketSendCommon -> PacketSerializeCommon
//       IssueOrderCore -> IssueOrderPacketBuilder -> PacketSendCommon -> PacketSerializeCommon
// ================================================================

namespace Offset {

// ================================================================
// GLOBAL POINTERS / INSTANCES  (all stable across hotfix)
// ================================================================
namespace Global {
    constexpr auto LocalPlayer      = 0x1DAB760;   // [D][P] local player ptr
    constexpr auto HeroManager      = 0x1D7A470;   // [D][P] hero list ptr
    constexpr auto GameTime         = 0x1D88580;   // [D][P] game time float
    constexpr auto MissileManager   = 0x1D7DD90;   // [D] missile manager ptr
    constexpr auto NavGrid          = 0x1D7DD08;   // [D] navigation grid ptr
    constexpr auto HudInstance      = 0x1D7A5B8;   // [D][P] HUD instance ptr
    constexpr auto UnderMouseObj    = 0x1D7DF90;   // [D] object under mouse cursor
    constexpr auto ViewPort         = 0x1D8D1F0;   // [D] viewport ptr
    constexpr auto ObjectManager    = 0x1D7A418;   // [D][P] object manager instance
    constexpr auto MinionManager    = 0x1D7A468;   // [IDA] minion+jungle list (CastSpellSafe decompile: qword_1D7A468)
    constexpr auto NetInstance      = 0x1D7A410;   // [IDA] net instance (Script-New had 0x1D7A3D0, new build +0x40)
    constexpr auto CursorInstance   = 0x1E056D8;   // [P] cursor position (Vec3)
    constexpr auto MouseScreenVec2  = 0x1D7DD38;   // [D] mouse 2D screen position
    constexpr auto ChatClient       = 0x1D8D240;   // [IDA] fallback, needs verification
    constexpr auto ChatInstance     = 0x1D7DFA0;   // [IDA] fallback
    constexpr auto r3dRenderer      = 0x1E3FEB8;   // [D] renderer instance (oViewPort2)
    constexpr auto ViewPort2        = 0x1E3FEB8;   // [D] viewport2/renderer
    constexpr auto MySpellState     = 0x1D80AE0;   // [D] spell state global
    constexpr auto TurretManager    = 0x1D870A8;   // [P] turret list
    constexpr auto ShopInstance     = 0x1D8D258;   // [IDA] fallback
    constexpr auto OpenWindowsArray = 0x1E3DC58;   // [IDA] fallback
    constexpr auto OpenWindowsCount = 0x1E3DC60;   // [IDA] fallback
}

// ================================================================
// FLAGS  (stable across hotfix - confirmed via decompile)
// ================================================================
namespace Flag {
    constexpr auto IssueOrderFlag   = 0x1CDDF88;   // [D][IDA] dword_1CDDF88 in IssueOrder (Chimera: order + 17)
    constexpr auto IssueOrder       = IssueOrderFlag; // Backward-compatible alias
    constexpr auto CastSpellFlag    = 0x1CDDF20;   // [D][IDA] byte_1CDDF20 in CastSpellSafe (Chimera CastSpellFlag)
    constexpr auto CastSpell        = CastSpellFlag; // Backward-compatible alias
}

// ================================================================
// FUNCTIONS (RVAs) — UPDATED for hotfix 2026-03-05
// ================================================================
namespace Function {
    // Core — 2026-03-12 hotfix (+0x10 from 03-05)
    constexpr auto IssueOrderCore       = 0x29FC20;     // [D] was 0x29FC10
    constexpr auto IssueOrder           = IssueOrderCore;
    constexpr auto IssueOrderPacketBuilder = 0x360CB0;  // Fallback (+0x10)
    constexpr auto IssueOrderPacketPostSend = 0x2CE8D0; // Fallback (+0x10)
    constexpr auto WorldToScreen        = 0x1241600;    // [D] was 0x1241370
    constexpr auto CastSpellWrapper     = 0x1E9A80;     // [D] was 0x1E9A70
    constexpr auto CastSpellSafe        = 0xBB9E00;     // [IDA] sub_BB9E00 (was 0xBB9E70 = MIDDLE of func!)
    constexpr auto CastSpellPacketA     = 0x91BF10;     // Fallback (+0x10)
    constexpr auto CastSpellPacketB     = 0x91B6C0;     // Fallback (+0x10)
    constexpr auto CastSpellPacketCharged = 0x91C7D0;   // Fallback (+0x10)
    constexpr auto PacketSendCommon     = 0x686940;     // Fallback (+0x10)
    constexpr auto PacketSerializeCommon = 0x686980;    // Fallback (+0x10)
    constexpr auto PrintChat            = 0x1095120;    // [P] (+0x10)
    constexpr auto GetBoundingRadius    = 0x285650;     // [D] was 0x285640
    constexpr auto GetAttackDelay       = 0x52C5A0;     // [D] was 0x52C590
    constexpr auto GetAttackWindup      = 0x52C4A0;     // [D] was 0x52C490
    constexpr auto GetCollisionFlags    = 0x1195E10;    // [D] was 0x1195B80
    constexpr auto GetPing              = 0x669EB0;     // [D] was 0x669F10

    // Object Iteration
    constexpr auto GetFirstObject       = 0x512920;     // [D] was 0x512910
    constexpr auto GetFirstObjectAlt    = 0x9D0410;     // [P] (+0x10)
    constexpr auto GetNextObject        = 0x513410;     // [D] was 0x513400
    constexpr auto FindObject           = 0x512110;     // [P] (+0x10)
    constexpr auto GetAiManager         = 0x50AAE0;     // [D] was 0x50AAD0
    constexpr auto GetAIManagerAlt      = 0x28D410;     // [P] (+0x10)
 
    // Type Checks
    constexpr auto IsTurret             = 0x308600;     // [D] was 0x3085F0
    constexpr auto IsHero               = 0x308700;     // [D] was 0x3086F0
    constexpr auto IsBuilding           = 0x308830;     // [P] (+0x10)
    constexpr auto IsAlive              = 0x2E6360;     // [D] was 0x2E6350
    constexpr auto IsDead               = 0x29B390;     // [P] (+0x10)
    constexpr auto IsTargetableByUnit   = 0x29E290;     // [P] (+0x10)
    constexpr auto IsVulnerable         = 0x29C050;     // [P] (+0x10)
    constexpr auto IsJungleMonster      = 0x29C220;     // [P] (+0x10)
    constexpr auto IsDragon             = 0x29B640;     // [P] (+0x10)
    constexpr auto IsElderDragon        = 0x29B6B0;     // [P] (+0x10)
    constexpr auto IsBaron              = 0x29AAA0;     // [P] (+0x10)
    constexpr auto IsSelectable         = 0x212180;     // [P] (+0x10)
    constexpr auto CompareTypeFlags     = 0x29CD40;     // [P] (+0x10)
    constexpr auto IsFleeing            = 0x20F340;     // [P] (+0x10)
    constexpr auto IsNoRender           = 0x20F390;     // [P] (+0x10)
    constexpr auto GetJungleType        = 0x66CE70;     // [P] (+0x10)
 
    // Attack / Combat
    constexpr auto CanAttack            = 0x1F90E0;     // [P] (+0x10)
    constexpr auto GetSpellCastInfo     = 0x283F20;     // [P] (+0x10)
    constexpr auto GetSpellSlot         = 0x90AA50;     // [P] (+0x10)
    constexpr auto GetResourceType      = 0x281240;     // [P] (+0x10)
    constexpr auto HasBuffOfType        = 0x296410;     // [P] (+0x10)
    constexpr auto GetGoldRedirectTgt   = 0x1FF9A0;     // [P] (+0x10)
 
    // Level Up
    constexpr auto LevelSpell           = 0xBA39C0;     // Fallback (+0x10)
 
    // Map / Minimap
    constexpr auto GetMapID             = 0x28E320;     // [D] was 0x28E310
 
    // Hooks / Callbacks
    constexpr auto OnCreateObject       = 0x517E20;     // [P] (+0x10)
    constexpr auto OnGameUpdate         = 0x5111C0;     // [P] (+0x10)
    constexpr auto OnProcessSpell       = 0x920590;     // [P] (+0x10)
    constexpr auto OnSpellImpact        = 0x917CA0;     // [P] (+0x10)
    constexpr auto OnStopCast           = 0x9208A0;     // [P] (+0x10)
    constexpr auto OnFinishCast         = 0x2C5770;     // [P] (+0x10)
    constexpr auto OnBuffAdd            = 0xBCDE90;     // [P] (+0x10)
    constexpr auto CreateClientEffect   = 0x869E90;     // [P] (+0x10)
}

// ================================================================
// GAME OBJECT STRUCT  (stable - struct offsets don't change)
// ================================================================
namespace GameObject {
    constexpr auto Index            = 0x10;         // [S]
    constexpr auto Team             = 0x3C;         // [S]
    constexpr auto Name             = 0x58;         // [S]
    constexpr auto NetId            = 0xCC;         // [D][S]
    constexpr auto Dead             = 0x250;        // [S]
    constexpr auto TeamAlt          = 0x259;        // [D]
    constexpr auto Position         = 0x25C;        // [S]
    constexpr auto EffectEmitter    = 0x258;        // [S]
    constexpr auto Visibility       = 0x2E0;        // [S]
    constexpr auto MissileClient    = 0x2D8;        // [S]
    constexpr auto Visible          = 0x308;        // [CE] verified: 0=fog, 1=visible on screen
    constexpr auto IsInvulnerable   = 0x5A0;        // [S]
    constexpr auto Radius           = 0x6F8;        // [D]
    constexpr auto RecallState      = 0xF48;        // [S]
    constexpr auto CharacterName    = 0x4330;       // [D]
    constexpr auto CharacterData    = 0x40C8;       // [S]
    constexpr auto Direction        = 0x21D8;       // [C] facing direction Vec3 (FaceDirection_s)
    constexpr auto ItemList         = 0x4D20;       // [C] array of 7 ItemSlot ptrs (6 items + trinket)
}

// ================================================================
// MANA
// ================================================================
namespace Mana {
    constexpr auto MP               = 0x360;        // [S]
    constexpr auto MaxMP            = 0x388;        // [S]
}

// ================================================================
// HEALTH (LeagueObfuscation<float>, 0x28 apart)
// ================================================================
namespace Health {
    constexpr auto HP               = 0x1080;       // [D]
    constexpr auto MaxHP            = 0x10A8;       // [D]
    constexpr auto HPMaxPenalty     = 0x10D0;       // [D]
    constexpr auto AllShield        = 0x1120;       // [D]
    constexpr auto PhysicalShield   = 0x1148;       // [D]
    constexpr auto MagicalShield    = 0x1170;       // [D]
    constexpr auto ChampSpecific    = 0x1198;       // [D]
    constexpr auto InHealAllied     = 0x11C0;       // [IDA] sub_2E3220: HP+320=0x1080+0x140
    constexpr auto InHealEnemy      = 0x11E8;       // [IDA] sub_2E3220: HP+360=0x1080+0x168
    constexpr auto InDamage         = 0x1210;       // [IDA] sub_2E3220: HP+400=0x1080+0x190
    constexpr auto StopShieldFade   = 0x1238;       // [IDA] sub_2E3220: HP+440=0x1080+0x1B8
}

// ================================================================
// TARGETABLE
// ================================================================
namespace Targetable {
    constexpr auto IsTargetable     = 0xED0;        // [D]
    constexpr auto TargetableFlags  = 0xEF8;        // [IDA] mIsTargetableToTeamFlags string xref
}

// ================================================================
// ACTION STATE
// ================================================================
namespace ActionState {
    constexpr auto State1           = 0x1470;       // [IDA] lea rdx,[rsi+1470h] -> sub_1FD490 "ActionState"
    constexpr auto State2           = 0x14A8;       // [IDA] 0x1470+0x38 -> sub_1FD490 "ActionState2"
}

// ================================================================
// DAMAGE MODIFIERS
// ================================================================
namespace DamageModifier {
    constexpr auto PhysDmgPercent   = 0x0E78;       // [IDA] lea rcx,[r14+0E78h] "mPhysicalDamagePercentageModifier"
    constexpr auto MagicDmgPercent  = 0x0EA0;       // [IDA] lea rcx,[r14+0EA0h] "mMagicalDamagePercentageModifier"
}

// ================================================================
// HERO STATS (LeagueObfuscation<float>, 0x28 apart)
// Stat block base: obj + 0x1B88
// ================================================================
namespace HeroStats {
    constexpr auto Base                     = 0x1B88;       // [D]

    // Cooldown / Ability Haste
    constexpr auto PercentCooldownMod       = 0x1B88;       // [D] base + 0x0
    constexpr auto AbilityHaste             = 0x1BB0;       // [D] base + 0x28
    constexpr auto PercentCooldownCapMod    = 0x1BD8;       // [D] base + 0x50
    constexpr auto PassiveCdEndTime         = 0x1C00;       // [D] base + 0x78
    constexpr auto PassiveCdTotalTime       = 0x1C28;       // [D] base + 0xA0
 
    // Minion-specific
    constexpr auto PercentDmgToBarracksMin  = 0x1C50;       // [D] base + 0xC8
    constexpr auto FlatDmgReducBarracks     = 0x1C78;       // [D] base + 0xF0
    constexpr auto IncreasedMoveSpeedMinion = 0x1CA0;       // [D] base + 0x118
 
    // Physical Damage
    constexpr auto FlatPhysicalDmgMod       = 0x1CC8;       // [D] base + 0x140
    constexpr auto PercentPhysicalDmgMod    = 0x1CF0;       // [D] base + 0x168
    constexpr auto PercentBonusPhysDmgMod   = 0x1D18;       // [D] base + 0x190
    constexpr auto PercentBasePhysDmgFlat   = 0x1D40;       // [D] base + 0x1B8
 
    // Magic Damage
    constexpr auto FlatMagicDmgMod          = 0x1D68;       // [D] base + 0x1E0
    constexpr auto PercentMagicDmgMod       = 0x1D90;       // [D] base + 0x208
    constexpr auto FlatMagicReduction       = 0x1DB8;       // [D] base + 0x230
    constexpr auto PercentMagicReduction    = 0x1DE0;       // [D] base + 0x258
 
    // Cast Range
    constexpr auto FlatCastRangeMod         = 0x1E08;       // [D] base + 0x280
 
    // Attack Speed
    constexpr auto AttackSpeedMod           = 0x1E30;       // [D] base + 0x2A8
    constexpr auto PercentAttackSpeedMod    = 0x1E58;       // [D] base + 0x2D0
    constexpr auto PercentMultiAtkSpeedMod  = 0x1E80;       // [D] base + 0x2F8
 
    // Healing
    constexpr auto PercentHealingAmountMod  = 0x1EA8;       // [D] base + 0x320
 
    // Attack Damage
    constexpr auto BaseAttackDamage         = 0x1ED0;       // [D] base + 0x348
    constexpr auto BaseAtkDmgSansScale      = 0x1EF8;       // [D] base + 0x370
    constexpr auto FlatBaseAtkDmgMod        = 0x1F20;       // [D] base + 0x398
    constexpr auto PercentBaseAtkDmgMod     = 0x1F48;       // [D] base + 0x3C0
 
    // Ability Power
    constexpr auto BaseAbilityDamage        = 0x1F70;       // [D] base + 0x3E8
 
    // Crit
    constexpr auto CritDamageMultiplier     = 0x1F98;       // [D] base + 0x410
    constexpr auto ScaleSkinCoef            = 0x1FC0;       // [D] base + 0x438
    constexpr auto Dodge                    = 0x1FE8;       // [D] base + 0x460
    constexpr auto Crit                     = 0x2010;       // [D] base + 0x488
 
    // Base HP Pool
    constexpr auto FlatBaseHPPoolMod        = 0x2038;       // [D] base + 0x4B0
 
    // Armor & MR
    // NOTE: Armor (0x2060) is TOTAL armor — already includes base + bonus.
    //       BonusArmor removed intentionally; use Armor directly for all calcs.
    constexpr auto Armor                    = 0x2060;       // [D] base + 0x4D8  (TOTAL armor — use this)
    // BonusArmor                           = 0x2088        // REMOVED — would double-count vs. Armor total
    constexpr auto SpellBlock               = 0x20B0;       // [D] base + 0x528  (MR, total)
    constexpr auto BonusSpellBlock          = 0x20D8;       // [D] base + 0x550
 
    // HP Regen
    constexpr auto HPRegenRate              = 0x2100;       // [D] base + 0x578
    constexpr auto BaseHPRegenRate          = 0x2128;       // [D] base + 0x5A0
 
    // Movement
    constexpr auto MoveSpeed                = 0x2150;       // [D] base + 0x5C8
    constexpr auto MoveSpeedBaseIncrease    = 0x2178;       // [D] base + 0x5F0
    constexpr auto AttackRange              = 0x21A0;       // [D] base + 0x618
 
    // Bubble Radius
    constexpr auto FlatBubbleRadiusMod      = 0x21C8;       // [D] base + 0x640
    constexpr auto PercentBubbleRadiusMod   = 0x21F0;       // [D] base + 0x668
 
    // Armor Penetration
    constexpr auto FlatArmorPen             = 0x2218;       // [D] base + 0x690
    constexpr auto PhysicalLethality        = 0x2240;       // [D] base + 0x6B8
    constexpr auto PercentArmorPen          = 0x2268;       // [D] base + 0x6E0
    constexpr auto PercentBonusArmorPen     = 0x2290;       // [D] base + 0x708
    constexpr auto PercentCritBonusArmorPen = 0x22B8;       // [D] base + 0x730
    constexpr auto PercentCritTotalArmorPen = 0x22E0;       // [D] base + 0x758
 
    // Magic Penetration
    constexpr auto FlatMagicPen             = 0x2308;       // [D] base + 0x780
    constexpr auto MagicLethality           = 0x2330;       // [D] base + 0x7A8
    constexpr auto PercentMagicPen          = 0x2358;       // [D] base + 0x7D0
    constexpr auto PercentBonusMagicPen     = 0x2380;       // [D] base + 0x7F8
 
    // Lifesteal / Vamp
    constexpr auto PercentLifeSteal         = 0x23A8;       // [D] base + 0x820
    constexpr auto PercentSpellVamp         = 0x23D0;       // [D] base + 0x848
    constexpr auto PercentOmnivamp          = 0x23F8;       // [D] base + 0x870
    constexpr auto PercentPhysicalVamp      = 0x2420;       // [D] base + 0x898
 
    // Pathing
    constexpr auto PathfindingRadiusMod     = 0x2448;       // [D] base + 0x8C0
 
    // Misc
    constexpr auto PercentCCReduction       = 0x2470;       // [D] base + 0x8E8
    constexpr auto PercentEXPBonus          = 0x2498;       // [D] base + 0x910
 
    // Base Armor/MR Flat Mods
    constexpr auto FlatBaseArmorMod         = 0x24C0;       // [D] base + 0x938
    constexpr auto FlatBaseSpellBlockMod    = 0x24E8;       // [D] base + 0x960
 
    // Resource Regen
    constexpr auto PARRegenRate             = 0x2510;       // [D] base + 0x988
    constexpr auto PrimaryARBaseRegenRate   = 0x2538;       // [D] base + 0x9B0
    constexpr auto SecondaryARRegenRate     = 0x2560;       // [D] base + 0x9D8
    constexpr auto SecondaryARBaseRegenRate = 0x2588;       // [D] base + 0xA00
 
    // Base Attack Speed
    constexpr auto FlatBaseAttackSpeedMod   = 0x25B0;       // [D] base + 0xA28
}

// ================================================================
// HERO-SPECIFIC
// ================================================================
namespace Hero {
    constexpr auto Gold                 = 0x2830;   // [D]
    constexpr auto GoldTotal            = 0x2858;   // [D]
    constexpr auto MinimumGold          = 0x2880;   // [D]
    constexpr auto FollowerTargetDelay  = 0x2DB8;   // [D] minion follower delay
    constexpr auto CombatType           = 0x2C98;   // [IDA] lea rdi,[r14+2C98h] "mCombatType"
    constexpr auto Exp                  = 0x4CF0;   // [D]
    constexpr auto LevelRef             = 0x4D18;   // [IDA] lea rcx,[r14+4D18h] "mLevelRef"
    constexpr auto LevelUpPoints        = 0x4D78;   // [chimera] LevelRef + 0x60 = skill points available
    constexpr auto VisionScore          = 0x55E0;   // [D]
    constexpr auto ShutdownValue        = 0x5608;   // [D]
    constexpr auto BaseGoldOnDeath      = 0x5630;   // [D]
    constexpr auto NeutralMinionsKilled = 0x5658;   // [IDA] lea rcx,[r14+5658h] "mNumNeutralMinionsKilled"
}

// ================================================================
// LIFETIME PROPS
// ================================================================
namespace Lifetime {
    constexpr auto Lifetime         = 0x0DB0;       // [IDA] lea rcx,[r14+0DB0h] "mLifetime"
    constexpr auto MaxLifetime      = 0x0DD8;       // [IDA] lea rcx,[r14+0DD8h] "mMaxLifetime"
    constexpr auto LifetimeTicks    = 0x0E00;       // [IDA] lea rcx,[r14+0E00h] "mLifetimeTicks"
}

// ================================================================
// SPELLBOOK & SPELL SLOTS
// ================================================================
namespace SpellBook {
    constexpr auto Offset           = 0x30E8;       // [D]
    constexpr auto SpellSlotArray   = 0xAE0;        // [D]
    constexpr auto ActiveSpellCast  = 0x3120;       // SpellBook::Offset + 0x38

    // SpellSlot (SpellDataInst)
    constexpr auto SlotLevel        = 0x28;         // [S]
    constexpr auto SlotCooldown     = 0x30;         // [S]
    constexpr auto SlotStacks       = 0x5C;         // [S]
    constexpr auto SlotTotalCd      = 0x74;         // [S]
    constexpr auto SlotSpellInput   = 0x120;        // [IDA] SpellInput/TargetClient (LOLDumper scans 0xB8 - wrong)
    constexpr auto SlotSpellInfo    = 0x128;        // [IDA] SpellInfo ptr (LOLDumper scans 0xC0 - wrong)
 
    // SpellInput
    constexpr auto InputTargetNetId = 0x14;         // [S]
    constexpr auto InputStartPos    = 0x18;         // [S]
    constexpr auto InputEndPos      = 0x24;         // [S]
 
    // SpellInfo
    constexpr auto InfoSpellData    = 0x60;         // [S]
 
    // SpellData
    constexpr auto DataSpellName    = 0x80;         // [S]
    constexpr auto SpellInfoNamePtr = 0x28;         // [brute confirmed] ptr -> char*
    constexpr auto DataManaCost     = 0x5F4;        // [S]
    constexpr auto DataResource     = 0x8;          // [D]
 
    // SpellData → SpellDataResource (SpellData + 0x60)
    constexpr auto DataResourceBase = 0x60;         // [IDA] SpellData+0x60 → SpellDataResource ptr
    constexpr auto ResCastRange     = 0x478;        // [C] array of 7 floats (per rank)
    constexpr auto ResMissileSpeed  = 0x518;        // [C] float missile speed
    constexpr auto ResLineWidth     = 0x568;        // [C] float line width
    constexpr auto ResMaxAmmo       = 0x3C0;        // [C] array of 7 ints (per rank)
    constexpr auto ResCastType      = 0x510;        // [C] targeting type enum
    constexpr auto ResMissileSpec   = 0x508;        // [C] missile specification ptr
    constexpr auto ResScriptName    = 0x80;         // [C] spell script name string
    constexpr auto ResCooldownTime  = 0x304;        // [C] array of 7 floats (per rank)
    constexpr auto ResAmmoRecharge  = 0x408;        // [C] array of 7 floats
    constexpr auto ResImgIconName   = 0x2A0;        // [C] icon name string
}

// ================================================================
// BUFF MANAGER
// ================================================================
namespace BuffManager {
    constexpr auto Offset           = 0x28B8;       // [D]
    constexpr auto EntriesEnd       = 0x10;         // [S]
    constexpr auto EntryBuff        = 0x10;         // [S]
    constexpr auto BuffType         = 0x0C;         // [S]
    constexpr auto BuffNamePtr      = 0x10;         // [S]
    constexpr auto BuffNameStr      = 0x8;          // [S]
    constexpr auto BuffStartTime    = 0x18;         // [S]
    constexpr auto BuffEndTime      = 0x1C;         // [S]
    constexpr auto BuffStacksAlt    = 0x38;         // [S]
    constexpr auto BuffStacks       = 0x78;         // [S]
}

// ================================================================
// AI MANAGER (Navigation / Pathing)
// ================================================================
namespace AiManager {
    constexpr auto Offset           = 0x41F0;       // [V] LeagueObfuscation offset from IDA sub_28E8C0
    constexpr auto InnerManager     = 0x10;         // [V] Final dereference to real AiManager
    constexpr auto NavPathPtr       = 0x30;         // [S] NavPath pointer (in dec struct)
    constexpr auto TargetPosition   = 0x034;        // [V] Vec3: Click destination / target position
    constexpr auto StartPath        = 0x88;         // [D]
    constexpr auto RefCount         = 0x1F0;        // [S]
    constexpr auto Velocity         = 0x318;        // [V] float: Movement speed value
    constexpr auto IsMoving         = 0x31C;        // [V] bool: Is currently moving
    constexpr auto CurrentSegment   = 0x320;        // [V] int: Current path segment index
    constexpr auto PathStart        = 0x330;        // [V] Vec3: Start of current path
    constexpr auto PathEnd          = 0x33C;        // [V] Vec3: End of current path
    constexpr auto Segments         = 0x348;        // [V] ptr: Waypoints array (Vec3[])
    constexpr auto NavArray         = 0x348;        // [V] ptr: Same as Segments (alias)
    constexpr auto SegmentsCount    = 0x350;        // [V] int: Number of waypoints
    constexpr auto HasPath          = 0x354;        // [V] int: Whether path data exists
    constexpr auto DashSpeed        = 0x360;        // [V] float: Dash speed
    constexpr auto IsDashing        = 0x384;        // [V] bool: Is currently dashing
    constexpr auto TargetPos2       = 0x3A8;        // [V] Vec3: Secondary target position
    constexpr auto ServerPos        = 0x474;        // [V] Vec3: Server-authoritative position
    constexpr auto MoveVec3         = 0x480;        // [S] Vec3: Move direction vector
}

// ================================================================
// HUD INSTANCE
// ================================================================
namespace Hud {
    constexpr auto Camera           = 0x18;         // [S]
    constexpr auto Input            = 0x28;         // [D] oHudMouse
    constexpr auto UserData         = 0x60;         // [S]
    constexpr auto SpellInfo        = 0x68;         // [D] oHudSpell

    // Camera / Zoom
    constexpr auto CameraZoom       = 0x324;        // [IDA] HudCamera + zoom offset
    constexpr auto CameraZoomLimits = 0x310;        // [IDA] ptr to zoom limits struct
    constexpr auto ZoomLimitsMin    = 0x24;         // [IDA] float min zoom in limits struct
    constexpr auto ZoomLimitsMax    = 0x28;         // [IDA] float max zoom in limits struct
    constexpr auto AltZoomLimits    = 0x3D0;        // [IDA] alternate zoom limits
    constexpr auto ZoomLockFlag1    = 0x344;        // [IDA] byte flag zoom lock 1
    constexpr auto ZoomLockFlag2    = 0x345;        // [IDA] byte flag zoom lock 2
 
    // Input / Cursor
    constexpr auto MouseWorldPos    = 0x34;         // [IDA] HudInput + mouse world pos
 
    // User Data
    constexpr auto SelectedObjNetId = 0x28;         // [S]
 
    // Chat  (ChatClient object offsets)
    constexpr auto ChatOpen         = 0x10;         // [IDA] byte flag: 1=chat input active, 0=closed (sub_3B4E00 sets ChatClient+16)
 
    // Viewport W2S
    constexpr auto ViewportW2S      = 0x2B0;        // [IDA] viewport W2S matrix offset
}

// ================================================================
// MISSILE OBJECT
// IDA MCP verified (2026-03-08):
//   sub_886AE0: missile init — copies CastInfo INLINE at missile+0x2C0
//   sub_845A50: CastInfo copy function (full struct layout mapped)
//   sub_90A0E0: missile collision — reads Position at +0x25C, CasterNetId at +0x358
//   sub_49E9F0: returns *(missile+0x128) = SpellData ptr
//   sub_28E710: returns*(missile+0x2C0) = first QWORD = SpellData ptr of CastInfo
//
// CastInfo is INLINE at missile+0x2C0 (NOT a pointer!)
// Read fields directly: startPos = Read<Vec3>(missile + StartPos)
// ================================================================
namespace Missile {
    // --- Missile Object (absolute offsets from missile base) ---
    constexpr auto SpellDataPtr     = 0x128;        // [IDA] sub_49E9F0: *(missile+0x128) = SpellData ptr
    constexpr auto Position         = 0x25C;        // [IDA] sub_90A0E0: Vec3 pos (inherited from GameObject)
    constexpr auto CastInfoBase     = 0x2C0;        // [IDA] sub_886AE0: CastInfo struct INLINE here (NOT a pointer!)
    constexpr auto MissileNetId     = 0x364;        // [IDA] sub_886AE0: [rsi+364h] = NetID (tree key) = CI+0xA4

    // --- CastInfo fields — ABSOLUTE offsets from missile base (0x2C0 + CI_*) ---
    //   Read directly: value = Read<T>(missile + offset)
    constexpr auto CI_SpellData     = 0x2C0;        // [IDA] QWORD: SpellData ptr (CastInfo+0x00)
    constexpr auto SpellName        = 0x2E0;        // [IDA] std::string SSO: spell name (CastInfo+0x20)
    constexpr auto MissileName      = 0x308;        // [IDA] std::string SSO: missile name (CastInfo+0x48)
    constexpr auto StartPos         = 0x388;        // [IDA] Vec3: start position (CastInfo+0xC8)
    constexpr auto EndPos           = 0x394;        // [IDA] Vec3: end position (CastInfo+0xD4)
    constexpr auto CastEndPos       = 0x3A4;        // [IDA] Vec3: cast end position (CastInfo+0xE4)
    constexpr auto CasterNetId      = 0x358;        // [IDA] int: source caster net id (CastInfo+0x98)
    constexpr auto TargetNetId      = 0x35C;        // [IDA] int: target net id (CastInfo+0x9C)
    constexpr auto CI_TargetNetId2  = 0x360;        // [IDA] int: secondary target (CastInfo+0xA0)
    constexpr auto CI_MissileNetId  = 0x364;        // [IDA] int: missile net id (CastInfo+0xA4)
 
    // --- CastInfo relative offsets (for code that needs CI base + offset pattern) ---
    constexpr auto CI_REL_SpellData    = 0x00;      // [IDA] CastInfo+0x00
    constexpr auto CI_REL_SpellName    = 0x20;      // [IDA] CastInfo+0x20
    constexpr auto CI_REL_MissileName  = 0x48;      // [IDA] CastInfo+0x48
    constexpr auto CI_REL_StartPos     = 0xC8;      // [IDA] CastInfo+0xC8
    constexpr auto CI_REL_EndPos       = 0xD4;      // [IDA] CastInfo+0xD4
    constexpr auto CI_REL_CastEndPos   = 0xE4;      // [IDA] CastInfo+0xE4
    constexpr auto CI_REL_CasterNetId  = 0x98;      // [IDA] CastInfo+0x98
    constexpr auto CI_REL_MissileNetId = 0xA4;      // [IDA] CastInfo+0xA4
 
    // --- Legacy aliases ---
    constexpr auto NetworkId        = MissileNetId; // 0x364
    constexpr auto SpellDataInst    = CI_SpellData; // 0x2C0
}

// ================================================================
// BASIC ATTACK / MISC
// ================================================================
namespace BasicAttack {
    constexpr auto Base             = 0x2C68;       // [D]
    constexpr auto Offset1          = 0x2C0;        // [D]
    constexpr auto Offset2          = 0x70;         // [D]
}

namespace Minion {
    constexpr auto LaneArray        = 0x68;         // [D] ptr to lane minion array (relative to MinionManager)
    constexpr auto LaneCount        = 0x70;         // [IDA] count of lane minions (relative to MinionManager)
    constexpr auto LaneType         = 0x4CC9;       // [CE] byte on obj: 4=Melee, 5=Ranged, 6=Cannon, 7=Super
}

// ================================================================
// DRAGON — Offsets for dragon soul detection (IDA sub_456A90 + sub_457DE0)
// ================================================================
namespace Dragon {
    constexpr auto CharacterHash    = 0x68;          // [IDA] DWORD hash on CharacterData (obj+CharData → +0x68)
    // Dragon Name Hash Table (global dword_1D995C0, 9 entries × 40 bytes)
    constexpr auto HashTable        = 0x1D995C0;     // [IDA] static hash table base
    constexpr auto HashTableEnd     = 0x1D99728;     // [IDA] end sentinel
    constexpr auto HashEntrySize    = 0x28;          // 40 bytes per entry (10 DWORDs)
    // Pre-computed dragon name hashes (sub_1074EA0 on dragon names)
    constexpr auto HashAir          = 0x11D34E07;    // SRU_Dragon_Air     → Cloud
    constexpr auto HashFire         = 0x99A9F7D9;    // SRU_Dragon_Fire    → Infernal
    constexpr auto HashWater        = 0x27F69DF4;    // SRU_Dragon_Water   → Ocean
    constexpr auto HashEarth        = 0x606D3187;    // SRU_Dragon_Earth   → Mountain
    constexpr auto HashHextech      = 0xA0808ACE;    // SRU_Dragon_Hextech → Hextech
    constexpr auto HashChemtech     = 0xF94EBA26;    // SRU_Dragon_Chemtech→ Chemtech
    constexpr auto HashRuined       = 0x518A146A;    // SRU_Dragon_Ruined  → Ruined
    constexpr auto HashElder        = 0x5944DC07;    // SRU_Dragon_Elder   → Elder
    constexpr auto HashParty        = 0x4B962AA3;    // SRU_Dragon_Party   → Party
}

// ================================================================
// SPELL CAST INFO (Active Spell)
// From: OnProcessSpell (0x920430) decompilation + chimera
// ================================================================
namespace SpellCastInfo {
    constexpr auto SpellData        = 0x0;          // [IDA] first QWORD = SpellData ptr
    constexpr auto SrcIndex         = 0x98;         // [C] source caster network index
    constexpr auto StartPos         = 0xD8;         // [C] Vec3 spell start position
    constexpr auto EndPos           = 0xE4;         // [C] Vec3 spell end position
    constexpr auto CastPos          = 0xF0;         // [C] Vec3 cast position
    constexpr auto TargetIndex      = 0x108;        // [C] target network index
    constexpr auto CastDelay        = 0x118;        // [C] float cast delay
    constexpr auto IsSpell          = 0x134;        // [C] bool is spell (not auto)
    constexpr auto IsSpecialAttack  = 0x13E;        // [C] bool is special attack
    constexpr auto IsAuto           = 0x141;        // [IDA] byte: is auto attack (chimera=0x13F)
    constexpr auto Slot             = 0x14C;        // [IDA] DWORD: spell slot index (chimera=0x148)
}

// ================================================================
// ITEM SYSTEM
// From: IDA MCP analysis + chimera_structures.h
// ================================================================
namespace ItemSystem {
    // GameObject::ItemList = 0x4D20 (in GameObject namespace)
    // Array of 7 ItemSlot pointers (6 items + trinket)
    constexpr auto SlotInfo         = 0x10;         // [IDA] ItemSlot+0x10 → ItemInfo ptr
    constexpr auto InfoData         = 0x38;         // [IDA] ItemInfo+0x38 → ItemData ptr
    constexpr auto InfoStacks       = 0x64;         // [C] ItemInfo+0x64 → stack count
    constexpr auto DataItemId       = 0xB4;         // [IDA] ItemData+0xB4 → item ID int
    constexpr auto DataAbilityHaste = 0x160;        // [C] ItemData stat
    constexpr auto DataHealth       = 0x164;        // [C] ItemData stat
    constexpr auto DataArmor        = 0x19C;        // [C] ItemData stat
    constexpr auto DataMR           = 0x1BC;        // [C] ItemData stat
    constexpr auto DataAD           = 0x1D8;        // [C] ItemData stat
    constexpr auto DataAP           = 0x1E0;        // [C] ItemData stat
    constexpr auto DataAtkSpeedMult = 0x20C;        // [C] ItemData stat
}

// ================================================================
// NAV GRID
// Source: sig 48 8B 05 ? ? ? ? 0F 28 DA → Global::NavGrid (0x1D7DD08)
// Chain: navGridPtr → +0x8 → NavGridManager → fields below
// IDA MCP verified (2026-03-11): decompile of GetCollisionFlags
// (0x1195B80), sub_1195BC0, sub_1190840, sub_119C040, sub_119C380,
// sub_119C210, sub_119C4F0 — all access *(qword_1D7DD08 + 8) = mgr
//
// KEY FIX: MinX/MinZ were WRONG (0x30/0x38).
// Decompile shows mgr[59] and mgr[61] → float at 59*4=0xEC, 61*4=0xF4
// This was causing intermittent bush/wall detection failure.
// ================================================================
namespace NavGrid {
    // Pointer chain
    constexpr auto NavGridMgr       = 0x8;          // [IDA] navGridPtr → +0x8 → manager

    // Map bounds (float)
    constexpr auto MinX             = 0xEC;         // [IDA] mgr[59] = world min X coordinate
    constexpr auto MinZ             = 0xF4;         // [IDA] mgr[61] = world min Z coordinate
    constexpr auto MaxX             = 0xF8;         // [IDA] mgr[62] = world max X coordinate
    constexpr auto MaxZ             = 0x100;        // [IDA] mgr[64] = world max Z coordinate
 
    // Cell data
    constexpr auto Data             = 0x110;        // [IDA] mgr+272 = ptr to cell array (16 bytes per cell)
    constexpr auto Width            = 0x708;        // [IDA] mgr+1800 = grid width (cells)
    constexpr auto Height           = 0x70C;        // [IDA] mgr+1804 = grid height (cells)
 
    // Scale
    constexpr auto InverseScale     = 0x714;        // [IDA] mgr+1812 = 1/cellSize (MULTIPLY to get cell index)
    constexpr auto Scale            = 0x710;        // [IDA] mgr[452] = cell size (used in bounds check)
 
    // Grass/Brush detection
    constexpr auto GrassRegions     = 0x158;        // [IDA] mgr+344 = grass region bitfield ptr
 
    // Cell structure: 16 bytes per cell
    // Layout: [uint64_t ptrData][uint16_t flags][uint16_t pad][uint32_t pad]
    // If ptrData != 0: real flags = *(uint16_t*)(ptrData + 6)
    // If ptrData == 0: real flags = cell.flags (at cell + 8)
    constexpr auto CellSize         = 16;           // [IDA] bytes per cell
 
    // Collision flag bitmask (from decompile of multiple functions)
    constexpr uint16_t FLAG_WALL    = 0x0001;       // [IDA] sub_119C380: bit 0 = wall
    constexpr uint16_t FLAG_NOWALK  = 0x0002;       // [IDA] sub_119C210: bit 1 = not walkable
    constexpr uint16_t FLAG_BRUSH   = 0x0C00;       // [IDA] sub_119C140: bits 10-11 = brush/grass
    constexpr uint16_t FLAG_SPECIAL = 0x1000;       // [IDA] sub_119C040: bit 12 = special terrain
}

// ================================================================
// MANAGER LIST
// ================================================================
namespace ManagerList {
    constexpr auto Items            = 0x8;          // [S]
    constexpr auto Size             = 0x10;         // [S]
}

// ================================================================
// MINIMAP
// ================================================================
namespace Minimap {
    constexpr auto MinimapParent    = 0x1D7A3D0;    // [CE] global ptr (same as NetInstance)
    constexpr auto MinimapHud       = 0x3B8;         // [CE] MinimapParent->+0x3B8 (was 0x288 in 14.23)
    constexpr auto HudVisible       = 0xD8;          // [CE] MinimapHud+0xD8 byte flag
}

// ================================================================
// EXTRA GLOBALS
// ================================================================
namespace Extra {
    constexpr auto TurretManager    = 0x1D87068;    // [P][IDA] 20 xrefs confirmed
    constexpr auto ViewMatrixInst   = 0x1E2C070;    // [P] view/projection matrix (from offsetplugin.hpp)
    constexpr auto IsClone          = 0x2BB2B0;     // [P] function RVA (+0x10)
}

// ================================================================
// VTABLES
// ================================================================
namespace VTable {
    constexpr auto AIMinionClient   = 0x18DD7F0;    // [P]
}

// ================================================================
// JUNGLE MONSTER NAME STRINGS
// These are string addresses in the binary - version specific!
// Found via IDA MCP find_regex on binary 0x2342000
// NOTE: These are for the IDA binary, NOT the dump binary!
//       For dump binary (0x202D000), re-scan needed.
// ================================================================
namespace JungleNames {
    // IDA binary (0x2342000) string addresses:
    constexpr auto SRU_RiftHerald   = 0x18d5358;    // [IDA] "SRU_RiftHerald"
    constexpr auto SRU_Horde        = 0x18d6690;    // [IDA] "SRU_Horde"
    constexpr auto SRU_Dragon       = 0x18d66B0;    // [IDA] "SRU_Dragon"
    constexpr auto SRU_Dragon_Elder = 0x18d66C0;    // [IDA] "SRU_Dragon_Elder"
    constexpr auto SRU_Baron        = 0x18e58D0;    // [IDA] "SRU_Baron"
}

// ================================================================
// OBJECT TYPE FLAGS (obfuscated field at obj+0x4C)
// Checked via CompareTypeFlags (sub_29CD30) — do NOT read directly!
// Use: Function::CompareTypeFlags(obj, FLAG_xxx)
// Found via IDA MCP decompile of sub_3088A0, sub_308B50, sub_3089A0, sub_308C70
// ================================================================
namespace TypeFlags {
    constexpr auto ObfuscatedField  = 0x4C;          // [IDA] obj+76 in sub_29CD30
    // Bit flags passed to CompareTypeFlags:
    constexpr auto Minion           = 0x0400;         // [IDA] sub_3089A0: flag 1024
    constexpr auto Hero             = 0x0800;         // [IDA] sub_308B50: flag 2048
    constexpr auto JungleMonster    = 0x2000;         // [IDA] sub_3088A0: flag 8192 (IsJungleMonster)
    constexpr auto LargeMonster     = 0x0080;         // [IDA] sub_345650: "Monster_Large" flag
    constexpr auto BuffMonster      = 0x0100;         // [IDA] sub_345650: "Monster_Buff" flag
    constexpr auto MinionSummon     = 0x0100;         // [IDA] sub_345650: "Minion_Summon" flag (same bit)
    constexpr auto Plant            = 0x8000;         // [IDA] sub_345650: "Plant" flag 32768
    constexpr auto CampMonster      = 0x10000;        // [IDA] sub_345650: 0x10000 after Plant
    constexpr auto Crab             = 0x2000;         // [IDA] sub_345650: "Monster_Crab" flag
    constexpr auto IsFleeing        = 0x0200;         // [IDA] sub_345650: fleeing check flag
    constexpr auto AttackableObj    = 0x0008;         // [IDA] sub_345650: attackable
    constexpr auto VisibleObj       = 0x0010;         // [IDA] sub_345650: visible flag
    constexpr auto RenderTarget     = 0x0020;         // [IDA] sub_345650: render target
    constexpr auto IsRecalling      = 0x4000;         // [IDA] sub_345650: recall check
    constexpr auto HasUltimate      = 0x20000;        // [IDA] sub_345650: vtable+2552 check
}

// ================================================================
// MINION CLASSIFICATION (from sub_BBB10 RegisterProperty table)
// LaneMinionType byte value on the minion object, registered via
// sub_10D1B80 with string name + numeric class ID
// Access: use GetJungleType (Function::GetJungleType) or read
//         the byte at the correct offset after finding it at runtime
// Found via IDA MCP decompile of sub_BBB10
// ================================================================
namespace MinionClass {
    // Class IDs (byte values):
    constexpr auto Unset            = 0;              // [IDA] v50=0 "Unset"
    constexpr auto Pet              = 1;              // [IDA] v54=1 "Pet"
    constexpr auto JungleMonster    = 2;              // [IDA] v58=2 "JungleMonster"
    constexpr auto TeamMinion       = 3;              // [IDA] v62=3 "TeamMinion"
    constexpr auto MeleeLaneMinion  = 4;              // [IDA] v66=4 "MeleeLaneMinion"
    constexpr auto RangedLaneMinion = 5;              // [IDA] v70=5 "RangedLaneMinion"
    constexpr auto SiegeLaneMinion  = 6;              // [IDA] v74=6 "SiegeLaneMinion"
    constexpr auto SuperLaneMinion  = 7;              // [IDA] v78=7 "SuperLaneMinion"
}

// ================================================================
// JUNGLE TYPE (from CharacterData sub-object)
// sub_345410 returns *(uint32_t*)(charData + 0x4A84)
// charData = obj + GameObject::CharacterData (0x40C8)
// GetJungleType (sub_66CE60) maps these to:
//   1 → type:0 (Normal),  2 → type:2 (Buff/Dragon), 3 → type:1 (Baron-like)
// Found via IDA MCP decompile of sub_345410 (returns charData+19076)
// ================================================================
namespace JungleType {
    constexpr auto TypeOffset       = 0x4A84;         // [IDA] charData + 19076 in sub_345410

    // Return values from GetJungleType:
    constexpr auto Normal           = 0;              // [IDA] sub_66CE60: case v23-1
    constexpr auto Baron            = 1;              // [IDA] sub_66CE60: v24==0 → return 1
    constexpr auto Dragon           = 2;              // [IDA] sub_66CE60: v22==0 → return 2
}

// ================================================================
// PLANT IDENTIFICATION
// Plants are identified via TypeFlags::Plant (0x8000)
// checked through CompareTypeFlags function
// Plant string names (IDA):
//   "Plant"             @ 0x18EF538
//   "OnPlantActivated"  @ 0x1902660
//   "AttackVisionplant" @ 0x18EBDA0
// Dragon subtypes (IDA string addresses):
//   SRU_Dragon_Air      @ 0x1908F78
//   SRU_Dragon_Fire     @ 0x1908F88
//   SRU_Dragon_Water    @ 0x1908F98
//   SRU_Dragon_Earth    @ 0x1908FB0
//   SRU_Dragon_Ruined   @ 0x1908FC8
//   SRU_Dragon_Hextech  @ 0x1908FE8
//   SRU_Dragon_Chemtech @ 0x1909000
//   SRU_Dragon_Party    @ 0x1909018
// ================================================================
namespace PlantInfo {
    // Plants are checked via: CompareTypeFlags(obj, TypeFlags::Plant)
    // Plant types are distinguished by CharacterName (obj + 0x4330):
    //   "SRU_Plant_Health"   → Honeyfruit (healing plant)
    //   "SRU_Plant_Satchel"  → Blast Cone (knockback plant)
    //   "SRU_Plant_Vision"   → Scryer's Bloom (vision plant)
}

} // namespace Offset
trankhanhtinh1 is offline

Old 15th March 2026, 05:40 AM   #12977
kral84
n00bie

kral84's Avatar

Join Date: Mar 2015
Posts: 9
Reputation: -120
Rep Power: 0
kral84 is an outcastkral84 is an outcast
Points: 7,965, Level: 10
Points: 7,965, Level: 10 Points: 7,965, Level: 10 Points: 7,965, Level: 10
Level up: 34%, 735 Points needed
Level up: 34% Level up: 34% Level up: 34%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
use version update for offset correct

The previous version was just a test, to find errors and make improvements. Now I can confirm that my LOLDUMPER dll can produce accurate results.

Code:
# pragma once
# include <cstdint>

// ================================================================
// League of Legends - Offsets
// Updated: 2026-03-12 (Hotfix) (LOLDumper v5.0 + offsetplugin.hpp + IDA MCP)
// Binary: League of Legends.exe
// Global/Function RVAs: from module size 0x202D000 (dump files)
// Struct offsets: verified via IDA on module size 0x2342000
// Base: 0x0 (relative offsets from module base)
//
// Sources:
//   [D]   = LOLDumper_full.h (pattern-scanned)
//   [P]   = offsetplugin.hpp (ida_lol_plugin.dll output)
//   [IDA] = IDA Pro MCP verified (decompile/disasm confirmed)
//   [CE]  = Cheat Engine verified at runtime
//   [S]   = struct offsets (unchanged between versions)
//   [C]   = chimera_structures.h reference (needs CE verify)
//
// Hotfix notes (2026-03-12):
//   - Function RVAs shifted +0x10 from 2026-03-05 hotfix
//   - All globals remained STABLE (confirmed via LOLDumper scan)
//   - Struct offsets STABLE (RegisterProperty-based, version-independent)
//   - LOLDumper re-scan confirmed globals unchanged
//   - Function deltas: IssueOrder +0x10, IsAlive +0x10, GetAttackDelay +0x10, etc.
//   - CastSpellSafe still at same RVA (offsetplugin.hpp: 0xBB9E60 → needs +0x10 verify)
//   - DetectionWatcher2 for Chimera-style mainloop_check is currently
//     resolved at runtime by signature: 4C 8B 3D ? ? ? ? 4D 85 FF 0F
//   - Current packet pipeline:
//       CastSpellSafe -> CastSpellPacketA/B/Charged -> PacketSendCommon -> PacketSerializeCommon
//       IssueOrderCore -> IssueOrderPacketBuilder -> PacketSendCommon -> PacketSerializeCommon
// ================================================================

namespace Offset {

// ================================================================
// GLOBAL POINTERS / INSTANCES  (all stable across hotfix)
// ================================================================
namespace Global {
    constexpr auto LocalPlayer      = 0x1DAB760;   // [D][P] local player ptr
    constexpr auto HeroManager      = 0x1D7A470;   // [D][P] hero list ptr
    constexpr auto GameTime         = 0x1D88580;   // [D][P] game time float
    constexpr auto MissileManager   = 0x1D7DD90;   // [D] missile manager ptr
    constexpr auto NavGrid          = 0x1D7DD08;   // [D] navigation grid ptr
    constexpr auto HudInstance      = 0x1D7A5B8;   // [D][P] HUD instance ptr
    constexpr auto UnderMouseObj    = 0x1D7DF90;   // [D] object under mouse cursor
    constexpr auto ViewPort         = 0x1D8D1F0;   // [D] viewport ptr
    constexpr auto ObjectManager    = 0x1D7A418;   // [D][P] object manager instance
    constexpr auto MinionManager    = 0x1D7A468;   // [IDA] minion+jungle list (CastSpellSafe decompile: qword_1D7A468)
    constexpr auto NetInstance      = 0x1D7A410;   // [IDA] net instance (Script-New had 0x1D7A3D0, new build +0x40)
    constexpr auto CursorInstance   = 0x1E056D8;   // [P] cursor position (Vec3)
    constexpr auto MouseScreenVec2  = 0x1D7DD38;   // [D] mouse 2D screen position
    constexpr auto ChatClient       = 0x1D8D240;   // [IDA] fallback, needs verification
    constexpr auto ChatInstance     = 0x1D7DFA0;   // [IDA] fallback
    constexpr auto r3dRenderer      = 0x1E3FEB8;   // [D] renderer instance (oViewPort2)
    constexpr auto ViewPort2        = 0x1E3FEB8;   // [D] viewport2/renderer
    constexpr auto MySpellState     = 0x1D80AE0;   // [D] spell state global
    constexpr auto TurretManager    = 0x1D870A8;   // [P] turret list
    constexpr auto ShopInstance     = 0x1D8D258;   // [IDA] fallback
    constexpr auto OpenWindowsArray = 0x1E3DC58;   // [IDA] fallback
    constexpr auto OpenWindowsCount = 0x1E3DC60;   // [IDA] fallback
}

// ================================================================
// FLAGS  (stable across hotfix - confirmed via decompile)
// ================================================================
namespace Flag {
    constexpr auto IssueOrderFlag   = 0x1CDDF88;   // [D][IDA] dword_1CDDF88 in IssueOrder (Chimera: order + 17)
    constexpr auto IssueOrder       = IssueOrderFlag; // Backward-compatible alias
    constexpr auto CastSpellFlag    = 0x1CDDF20;   // [D][IDA] byte_1CDDF20 in CastSpellSafe (Chimera CastSpellFlag)
    constexpr auto CastSpell        = CastSpellFlag; // Backward-compatible alias
}

// ================================================================
// FUNCTIONS (RVAs) — UPDATED for hotfix 2026-03-05
// ================================================================
namespace Function {
    // Core — 2026-03-12 hotfix (+0x10 from 03-05)
    constexpr auto IssueOrderCore       = 0x29FC20;     // [D] was 0x29FC10
    constexpr auto IssueOrder           = IssueOrderCore;
    constexpr auto IssueOrderPacketBuilder = 0x360CB0;  // Fallback (+0x10)
    constexpr auto IssueOrderPacketPostSend = 0x2CE8D0; // Fallback (+0x10)
    constexpr auto WorldToScreen        = 0x1241600;    // [D] was 0x1241370
    constexpr auto CastSpellWrapper     = 0x1E9A80;     // [D] was 0x1E9A70
    constexpr auto CastSpellSafe        = 0xBB9E00;     // [IDA] sub_BB9E00 (was 0xBB9E70 = MIDDLE of func!)
    constexpr auto CastSpellPacketA     = 0x91BF10;     // Fallback (+0x10)
    constexpr auto CastSpellPacketB     = 0x91B6C0;     // Fallback (+0x10)
    constexpr auto CastSpellPacketCharged = 0x91C7D0;   // Fallback (+0x10)
    constexpr auto PacketSendCommon     = 0x686940;     // Fallback (+0x10)
    constexpr auto PacketSerializeCommon = 0x686980;    // Fallback (+0x10)
    constexpr auto PrintChat            = 0x1095120;    // [P] (+0x10)
    constexpr auto GetBoundingRadius    = 0x285650;     // [D] was 0x285640
    constexpr auto GetAttackDelay       = 0x52C5A0;     // [D] was 0x52C590
    constexpr auto GetAttackWindup      = 0x52C4A0;     // [D] was 0x52C490
    constexpr auto GetCollisionFlags    = 0x1195E10;    // [D] was 0x1195B80
    constexpr auto GetPing              = 0x669EB0;     // [D] was 0x669F10

    // Object Iteration
    constexpr auto GetFirstObject       = 0x512920;     // [D] was 0x512910
    constexpr auto GetFirstObjectAlt    = 0x9D0410;     // [P] (+0x10)
    constexpr auto GetNextObject        = 0x513410;     // [D] was 0x513400
    constexpr auto FindObject           = 0x512110;     // [P] (+0x10)
    constexpr auto GetAiManager         = 0x50AAE0;     // [D] was 0x50AAD0
    constexpr auto GetAIManagerAlt      = 0x28D410;     // [P] (+0x10)
 
    // Type Checks
    constexpr auto IsTurret             = 0x308600;     // [D] was 0x3085F0
    constexpr auto IsHero               = 0x308700;     // [D] was 0x3086F0
    constexpr auto IsBuilding           = 0x308830;     // [P] (+0x10)
    constexpr auto IsAlive              = 0x2E6360;     // [D] was 0x2E6350
    constexpr auto IsDead               = 0x29B390;     // [P] (+0x10)
    constexpr auto IsTargetableByUnit   = 0x29E290;     // [P] (+0x10)
    constexpr auto IsVulnerable         = 0x29C050;     // [P] (+0x10)
    constexpr auto IsJungleMonster      = 0x29C220;     // [P] (+0x10)
    constexpr auto IsDragon             = 0x29B640;     // [P] (+0x10)
    constexpr auto IsElderDragon        = 0x29B6B0;     // [P] (+0x10)
    constexpr auto IsBaron              = 0x29AAA0;     // [P] (+0x10)
    constexpr auto IsSelectable         = 0x212180;     // [P] (+0x10)
    constexpr auto CompareTypeFlags     = 0x29CD40;     // [P] (+0x10)
    constexpr auto IsFleeing            = 0x20F340;     // [P] (+0x10)
    constexpr auto IsNoRender           = 0x20F390;     // [P] (+0x10)
    constexpr auto GetJungleType        = 0x66CE70;     // [P] (+0x10)
 
    // Attack / Combat
    constexpr auto CanAttack            = 0x1F90E0;     // [P] (+0x10)
    constexpr auto GetSpellCastInfo     = 0x283F20;     // [P] (+0x10)
    constexpr auto GetSpellSlot         = 0x90AA50;     // [P] (+0x10)
    constexpr auto GetResourceType      = 0x281240;     // [P] (+0x10)
    constexpr auto HasBuffOfType        = 0x296410;     // [P] (+0x10)
    constexpr auto GetGoldRedirectTgt   = 0x1FF9A0;     // [P] (+0x10)
 
    // Level Up
    constexpr auto LevelSpell           = 0xBA39C0;     // Fallback (+0x10)
 
    // Map / Minimap
    constexpr auto GetMapID             = 0x28E320;     // [D] was 0x28E310
 
    // Hooks / Callbacks
    constexpr auto OnCreateObject       = 0x517E20;     // [P] (+0x10)
    constexpr auto OnGameUpdate         = 0x5111C0;     // [P] (+0x10)
    constexpr auto OnProcessSpell       = 0x920590;     // [P] (+0x10)
    constexpr auto OnSpellImpact        = 0x917CA0;     // [P] (+0x10)
    constexpr auto OnStopCast           = 0x9208A0;     // [P] (+0x10)
    constexpr auto OnFinishCast         = 0x2C5770;     // [P] (+0x10)
    constexpr auto OnBuffAdd            = 0xBCDE90;     // [P] (+0x10)
    constexpr auto CreateClientEffect   = 0x869E90;     // [P] (+0x10)
}

// ================================================================
// GAME OBJECT STRUCT  (stable - struct offsets don't change)
// ================================================================
namespace GameObject {
    constexpr auto Index            = 0x10;         // [S]
    constexpr auto Team             = 0x3C;         // [S]
    constexpr auto Name             = 0x58;         // [S]
    constexpr auto NetId            = 0xCC;         // [D][S]
    constexpr auto Dead             = 0x250;        // [S]
    constexpr auto TeamAlt          = 0x259;        // [D]
    constexpr auto Position         = 0x25C;        // [S]
    constexpr auto EffectEmitter    = 0x258;        // [S]
    constexpr auto Visibility       = 0x2E0;        // [S]
    constexpr auto MissileClient    = 0x2D8;        // [S]
    constexpr auto Visible          = 0x308;        // [CE] verified: 0=fog, 1=visible on screen
    constexpr auto IsInvulnerable   = 0x5A0;        // [S]
    constexpr auto Radius           = 0x6F8;        // [D]
    constexpr auto RecallState      = 0xF48;        // [S]
    constexpr auto CharacterName    = 0x4330;       // [D]
    constexpr auto CharacterData    = 0x40C8;       // [S]
    constexpr auto Direction        = 0x21D8;       // [C] facing direction Vec3 (FaceDirection_s)
    constexpr auto ItemList         = 0x4D20;       // [C] array of 7 ItemSlot ptrs (6 items + trinket)
}

// ================================================================
// MANA
// ================================================================
namespace Mana {
    constexpr auto MP               = 0x360;        // [S]
    constexpr auto MaxMP            = 0x388;        // [S]
}

// ================================================================
// HEALTH (LeagueObfuscation<float>, 0x28 apart)
// ================================================================
namespace Health {
    constexpr auto HP               = 0x1080;       // [D]
    constexpr auto MaxHP            = 0x10A8;       // [D]
    constexpr auto HPMaxPenalty     = 0x10D0;       // [D]
    constexpr auto AllShield        = 0x1120;       // [D]
    constexpr auto PhysicalShield   = 0x1148;       // [D]
    constexpr auto MagicalShield    = 0x1170;       // [D]
    constexpr auto ChampSpecific    = 0x1198;       // [D]
    constexpr auto InHealAllied     = 0x11C0;       // [IDA] sub_2E3220: HP+320=0x1080+0x140
    constexpr auto InHealEnemy      = 0x11E8;       // [IDA] sub_2E3220: HP+360=0x1080+0x168
    constexpr auto InDamage         = 0x1210;       // [IDA] sub_2E3220: HP+400=0x1080+0x190
    constexpr auto StopShieldFade   = 0x1238;       // [IDA] sub_2E3220: HP+440=0x1080+0x1B8
}

// ================================================================
// TARGETABLE
// ================================================================
namespace Targetable {
    constexpr auto IsTargetable     = 0xED0;        // [D]
    constexpr auto TargetableFlags  = 0xEF8;        // [IDA] mIsTargetableToTeamFlags string xref
}

// ================================================================
// ACTION STATE
// ================================================================
namespace ActionState {
    constexpr auto State1           = 0x1470;       // [IDA] lea rdx,[rsi+1470h] -> sub_1FD490 "ActionState"
    constexpr auto State2           = 0x14A8;       // [IDA] 0x1470+0x38 -> sub_1FD490 "ActionState2"
}

// ================================================================
// DAMAGE MODIFIERS
// ================================================================
namespace DamageModifier {
    constexpr auto PhysDmgPercent   = 0x0E78;       // [IDA] lea rcx,[r14+0E78h] "mPhysicalDamagePercentageModifier"
    constexpr auto MagicDmgPercent  = 0x0EA0;       // [IDA] lea rcx,[r14+0EA0h] "mMagicalDamagePercentageModifier"
}

// ================================================================
// HERO STATS (LeagueObfuscation<float>, 0x28 apart)
// Stat block base: obj + 0x1B88
// ================================================================
namespace HeroStats {
    constexpr auto Base                     = 0x1B88;       // [D]

    // Cooldown / Ability Haste
    constexpr auto PercentCooldownMod       = 0x1B88;       // [D] base + 0x0
    constexpr auto AbilityHaste             = 0x1BB0;       // [D] base + 0x28
    constexpr auto PercentCooldownCapMod    = 0x1BD8;       // [D] base + 0x50
    constexpr auto PassiveCdEndTime         = 0x1C00;       // [D] base + 0x78
    constexpr auto PassiveCdTotalTime       = 0x1C28;       // [D] base + 0xA0
 
    // Minion-specific
    constexpr auto PercentDmgToBarracksMin  = 0x1C50;       // [D] base + 0xC8
    constexpr auto FlatDmgReducBarracks     = 0x1C78;       // [D] base + 0xF0
    constexpr auto IncreasedMoveSpeedMinion = 0x1CA0;       // [D] base + 0x118
 
    // Physical Damage
    constexpr auto FlatPhysicalDmgMod       = 0x1CC8;       // [D] base + 0x140
    constexpr auto PercentPhysicalDmgMod    = 0x1CF0;       // [D] base + 0x168
    constexpr auto PercentBonusPhysDmgMod   = 0x1D18;       // [D] base + 0x190
    constexpr auto PercentBasePhysDmgFlat   = 0x1D40;       // [D] base + 0x1B8
 
    // Magic Damage
    constexpr auto FlatMagicDmgMod          = 0x1D68;       // [D] base + 0x1E0
    constexpr auto PercentMagicDmgMod       = 0x1D90;       // [D] base + 0x208
    constexpr auto FlatMagicReduction       = 0x1DB8;       // [D] base + 0x230
    constexpr auto PercentMagicReduction    = 0x1DE0;       // [D] base + 0x258
 
    // Cast Range
    constexpr auto FlatCastRangeMod         = 0x1E08;       // [D] base + 0x280
 
    // Attack Speed
    constexpr auto AttackSpeedMod           = 0x1E30;       // [D] base + 0x2A8
    constexpr auto PercentAttackSpeedMod    = 0x1E58;       // [D] base + 0x2D0
    constexpr auto PercentMultiAtkSpeedMod  = 0x1E80;       // [D] base + 0x2F8
 
    // Healing
    constexpr auto PercentHealingAmountMod  = 0x1EA8;       // [D] base + 0x320
 
    // Attack Damage
    constexpr auto BaseAttackDamage         = 0x1ED0;       // [D] base + 0x348
    constexpr auto BaseAtkDmgSansScale      = 0x1EF8;       // [D] base + 0x370
    constexpr auto FlatBaseAtkDmgMod        = 0x1F20;       // [D] base + 0x398
    constexpr auto PercentBaseAtkDmgMod     = 0x1F48;       // [D] base + 0x3C0
 
    // Ability Power
    constexpr auto BaseAbilityDamage        = 0x1F70;       // [D] base + 0x3E8
 
    // Crit
    constexpr auto CritDamageMultiplier     = 0x1F98;       // [D] base + 0x410
    constexpr auto ScaleSkinCoef            = 0x1FC0;       // [D] base + 0x438
    constexpr auto Dodge                    = 0x1FE8;       // [D] base + 0x460
    constexpr auto Crit                     = 0x2010;       // [D] base + 0x488
 
    // Base HP Pool
    constexpr auto FlatBaseHPPoolMod        = 0x2038;       // [D] base + 0x4B0
 
    // Armor & MR
    // NOTE: Armor (0x2060) is TOTAL armor — already includes base + bonus.
    //       BonusArmor removed intentionally; use Armor directly for all calcs.
    constexpr auto Armor                    = 0x2060;       // [D] base + 0x4D8  (TOTAL armor — use this)
    // BonusArmor                           = 0x2088        // REMOVED — would double-count vs. Armor total
    constexpr auto SpellBlock               = 0x20B0;       // [D] base + 0x528  (MR, total)
    constexpr auto BonusSpellBlock          = 0x20D8;       // [D] base + 0x550
 
    // HP Regen
    constexpr auto HPRegenRate              = 0x2100;       // [D] base + 0x578
    constexpr auto BaseHPRegenRate          = 0x2128;       // [D] base + 0x5A0
 
    // Movement
    constexpr auto MoveSpeed                = 0x2150;       // [D] base + 0x5C8
    constexpr auto MoveSpeedBaseIncrease    = 0x2178;       // [D] base + 0x5F0
    constexpr auto AttackRange              = 0x21A0;       // [D] base + 0x618
 
    // Bubble Radius
    constexpr auto FlatBubbleRadiusMod      = 0x21C8;       // [D] base + 0x640
    constexpr auto PercentBubbleRadiusMod   = 0x21F0;       // [D] base + 0x668
 
    // Armor Penetration
    constexpr auto FlatArmorPen             = 0x2218;       // [D] base + 0x690
    constexpr auto PhysicalLethality        = 0x2240;       // [D] base + 0x6B8
    constexpr auto PercentArmorPen          = 0x2268;       // [D] base + 0x6E0
    constexpr auto PercentBonusArmorPen     = 0x2290;       // [D] base + 0x708
    constexpr auto PercentCritBonusArmorPen = 0x22B8;       // [D] base + 0x730
    constexpr auto PercentCritTotalArmorPen = 0x22E0;       // [D] base + 0x758
 
    // Magic Penetration
    constexpr auto FlatMagicPen             = 0x2308;       // [D] base + 0x780
    constexpr auto MagicLethality           = 0x2330;       // [D] base + 0x7A8
    constexpr auto PercentMagicPen          = 0x2358;       // [D] base + 0x7D0
    constexpr auto PercentBonusMagicPen     = 0x2380;       // [D] base + 0x7F8
 
    // Lifesteal / Vamp
    constexpr auto PercentLifeSteal         = 0x23A8;       // [D] base + 0x820
    constexpr auto PercentSpellVamp         = 0x23D0;       // [D] base + 0x848
    constexpr auto PercentOmnivamp          = 0x23F8;       // [D] base + 0x870
    constexpr auto PercentPhysicalVamp      = 0x2420;       // [D] base + 0x898
 
    // Pathing
    constexpr auto PathfindingRadiusMod     = 0x2448;       // [D] base + 0x8C0
 
    // Misc
    constexpr auto PercentCCReduction       = 0x2470;       // [D] base + 0x8E8
    constexpr auto PercentEXPBonus          = 0x2498;       // [D] base + 0x910
 
    // Base Armor/MR Flat Mods
    constexpr auto FlatBaseArmorMod         = 0x24C0;       // [D] base + 0x938
    constexpr auto FlatBaseSpellBlockMod    = 0x24E8;       // [D] base + 0x960
 
    // Resource Regen
    constexpr auto PARRegenRate             = 0x2510;       // [D] base + 0x988
    constexpr auto PrimaryARBaseRegenRate   = 0x2538;       // [D] base + 0x9B0
    constexpr auto SecondaryARRegenRate     = 0x2560;       // [D] base + 0x9D8
    constexpr auto SecondaryARBaseRegenRate = 0x2588;       // [D] base + 0xA00
 
    // Base Attack Speed
    constexpr auto FlatBaseAttackSpeedMod   = 0x25B0;       // [D] base + 0xA28
}

// ================================================================
// HERO-SPECIFIC
// ================================================================
namespace Hero {
    constexpr auto Gold                 = 0x2830;   // [D]
    constexpr auto GoldTotal            = 0x2858;   // [D]
    constexpr auto MinimumGold          = 0x2880;   // [D]
    constexpr auto FollowerTargetDelay  = 0x2DB8;   // [D] minion follower delay
    constexpr auto CombatType           = 0x2C98;   // [IDA] lea rdi,[r14+2C98h] "mCombatType"
    constexpr auto Exp                  = 0x4CF0;   // [D]
    constexpr auto LevelRef             = 0x4D18;   // [IDA] lea rcx,[r14+4D18h] "mLevelRef"
    constexpr auto LevelUpPoints        = 0x4D78;   // [chimera] LevelRef + 0x60 = skill points available
    constexpr auto VisionScore          = 0x55E0;   // [D]
    constexpr auto ShutdownValue        = 0x5608;   // [D]
    constexpr auto BaseGoldOnDeath      = 0x5630;   // [D]
    constexpr auto NeutralMinionsKilled = 0x5658;   // [IDA] lea rcx,[r14+5658h] "mNumNeutralMinionsKilled"
}

// ================================================================
// LIFETIME PROPS
// ================================================================
namespace Lifetime {
    constexpr auto Lifetime         = 0x0DB0;       // [IDA] lea rcx,[r14+0DB0h] "mLifetime"
    constexpr auto MaxLifetime      = 0x0DD8;       // [IDA] lea rcx,[r14+0DD8h] "mMaxLifetime"
    constexpr auto LifetimeTicks    = 0x0E00;       // [IDA] lea rcx,[r14+0E00h] "mLifetimeTicks"
}

// ================================================================
// SPELLBOOK & SPELL SLOTS
// ================================================================
namespace SpellBook {
    constexpr auto Offset           = 0x30E8;       // [D]
    constexpr auto SpellSlotArray   = 0xAE0;        // [D]
    constexpr auto ActiveSpellCast  = 0x3120;       // SpellBook::Offset + 0x38

    // SpellSlot (SpellDataInst)
    constexpr auto SlotLevel        = 0x28;         // [S]
    constexpr auto SlotCooldown     = 0x30;         // [S]
    constexpr auto SlotStacks       = 0x5C;         // [S]
    constexpr auto SlotTotalCd      = 0x74;         // [S]
    constexpr auto SlotSpellInput   = 0x120;        // [IDA] SpellInput/TargetClient (LOLDumper scans 0xB8 - wrong)
    constexpr auto SlotSpellInfo    = 0x128;        // [IDA] SpellInfo ptr (LOLDumper scans 0xC0 - wrong)
 
    // SpellInput
    constexpr auto InputTargetNetId = 0x14;         // [S]
    constexpr auto InputStartPos    = 0x18;         // [S]
    constexpr auto InputEndPos      = 0x24;         // [S]
 
    // SpellInfo
    constexpr auto InfoSpellData    = 0x60;         // [S]
 
    // SpellData
    constexpr auto DataSpellName    = 0x80;         // [S]
    constexpr auto SpellInfoNamePtr = 0x28;         // [brute confirmed] ptr -> char*
    constexpr auto DataManaCost     = 0x5F4;        // [S]
    constexpr auto DataResource     = 0x8;          // [D]
 
    // SpellData → SpellDataResource (SpellData + 0x60)
    constexpr auto DataResourceBase = 0x60;         // [IDA] SpellData+0x60 → SpellDataResource ptr
    constexpr auto ResCastRange     = 0x478;        // [C] array of 7 floats (per rank)
    constexpr auto ResMissileSpeed  = 0x518;        // [C] float missile speed
    constexpr auto ResLineWidth     = 0x568;        // [C] float line width
    constexpr auto ResMaxAmmo       = 0x3C0;        // [C] array of 7 ints (per rank)
    constexpr auto ResCastType      = 0x510;        // [C] targeting type enum
    constexpr auto ResMissileSpec   = 0x508;        // [C] missile specification ptr
    constexpr auto ResScriptName    = 0x80;         // [C] spell script name string
    constexpr auto ResCooldownTime  = 0x304;        // [C] array of 7 floats (per rank)
    constexpr auto ResAmmoRecharge  = 0x408;        // [C] array of 7 floats
    constexpr auto ResImgIconName   = 0x2A0;        // [C] icon name string
}

// ================================================================
// BUFF MANAGER
// ================================================================
namespace BuffManager {
    constexpr auto Offset           = 0x28B8;       // [D]
    constexpr auto EntriesEnd       = 0x10;         // [S]
    constexpr auto EntryBuff        = 0x10;         // [S]
    constexpr auto BuffType         = 0x0C;         // [S]
    constexpr auto BuffNamePtr      = 0x10;         // [S]
    constexpr auto BuffNameStr      = 0x8;          // [S]
    constexpr auto BuffStartTime    = 0x18;         // [S]
    constexpr auto BuffEndTime      = 0x1C;         // [S]
    constexpr auto BuffStacksAlt    = 0x38;         // [S]
    constexpr auto BuffStacks       = 0x78;         // [S]
}

// ================================================================
// AI MANAGER (Navigation / Pathing)
// ================================================================
namespace AiManager {
    constexpr auto Offset           = 0x41F0;       // [V] LeagueObfuscation offset from IDA sub_28E8C0
    constexpr auto InnerManager     = 0x10;         // [V] Final dereference to real AiManager
    constexpr auto NavPathPtr       = 0x30;         // [S] NavPath pointer (in dec struct)
    constexpr auto TargetPosition   = 0x034;        // [V] Vec3: Click destination / target position
    constexpr auto StartPath        = 0x88;         // [D]
    constexpr auto RefCount         = 0x1F0;        // [S]
    constexpr auto Velocity         = 0x318;        // [V] float: Movement speed value
    constexpr auto IsMoving         = 0x31C;        // [V] bool: Is currently moving
    constexpr auto CurrentSegment   = 0x320;        // [V] int: Current path segment index
    constexpr auto PathStart        = 0x330;        // [V] Vec3: Start of current path
    constexpr auto PathEnd          = 0x33C;        // [V] Vec3: End of current path
    constexpr auto Segments         = 0x348;        // [V] ptr: Waypoints array (Vec3[])
    constexpr auto NavArray         = 0x348;        // [V] ptr: Same as Segments (alias)
    constexpr auto SegmentsCount    = 0x350;        // [V] int: Number of waypoints
    constexpr auto HasPath          = 0x354;        // [V] int: Whether path data exists
    constexpr auto DashSpeed        = 0x360;        // [V] float: Dash speed
    constexpr auto IsDashing        = 0x384;        // [V] bool: Is currently dashing
    constexpr auto TargetPos2       = 0x3A8;        // [V] Vec3: Secondary target position
    constexpr auto ServerPos        = 0x474;        // [V] Vec3: Server-authoritative position
    constexpr auto MoveVec3         = 0x480;        // [S] Vec3: Move direction vector
}

// ================================================================
// HUD INSTANCE
// ================================================================
namespace Hud {
    constexpr auto Camera           = 0x18;         // [S]
    constexpr auto Input            = 0x28;         // [D] oHudMouse
    constexpr auto UserData         = 0x60;         // [S]
    constexpr auto SpellInfo        = 0x68;         // [D] oHudSpell

    // Camera / Zoom
    constexpr auto CameraZoom       = 0x324;        // [IDA] HudCamera + zoom offset
    constexpr auto CameraZoomLimits = 0x310;        // [IDA] ptr to zoom limits struct
    constexpr auto ZoomLimitsMin    = 0x24;         // [IDA] float min zoom in limits struct
    constexpr auto ZoomLimitsMax    = 0x28;         // [IDA] float max zoom in limits struct
    constexpr auto AltZoomLimits    = 0x3D0;        // [IDA] alternate zoom limits
    constexpr auto ZoomLockFlag1    = 0x344;        // [IDA] byte flag zoom lock 1
    constexpr auto ZoomLockFlag2    = 0x345;        // [IDA] byte flag zoom lock 2
 
    // Input / Cursor
    constexpr auto MouseWorldPos    = 0x34;         // [IDA] HudInput + mouse world pos
 
    // User Data
    constexpr auto SelectedObjNetId = 0x28;         // [S]
 
    // Chat  (ChatClient object offsets)
    constexpr auto ChatOpen         = 0x10;         // [IDA] byte flag: 1=chat input active, 0=closed (sub_3B4E00 sets ChatClient+16)
 
    // Viewport W2S
    constexpr auto ViewportW2S      = 0x2B0;        // [IDA] viewport W2S matrix offset
}

// ================================================================
// MISSILE OBJECT
// IDA MCP verified (2026-03-08):
//   sub_886AE0: missile init — copies CastInfo INLINE at missile+0x2C0
//   sub_845A50: CastInfo copy function (full struct layout mapped)
//   sub_90A0E0: missile collision — reads Position at +0x25C, CasterNetId at +0x358
//   sub_49E9F0: returns *(missile+0x128) = SpellData ptr
//   sub_28E710: returns*(missile+0x2C0) = first QWORD = SpellData ptr of CastInfo
//
// CastInfo is INLINE at missile+0x2C0 (NOT a pointer!)
// Read fields directly: startPos = Read<Vec3>(missile + StartPos)
// ================================================================
namespace Missile {
    // --- Missile Object (absolute offsets from missile base) ---
    constexpr auto SpellDataPtr     = 0x128;        // [IDA] sub_49E9F0: *(missile+0x128) = SpellData ptr
    constexpr auto Position         = 0x25C;        // [IDA] sub_90A0E0: Vec3 pos (inherited from GameObject)
    constexpr auto CastInfoBase     = 0x2C0;        // [IDA] sub_886AE0: CastInfo struct INLINE here (NOT a pointer!)
    constexpr auto MissileNetId     = 0x364;        // [IDA] sub_886AE0: [rsi+364h] = NetID (tree key) = CI+0xA4

    // --- CastInfo fields — ABSOLUTE offsets from missile base (0x2C0 + CI_*) ---
    //   Read directly: value = Read<T>(missile + offset)
    constexpr auto CI_SpellData     = 0x2C0;        // [IDA] QWORD: SpellData ptr (CastInfo+0x00)
    constexpr auto SpellName        = 0x2E0;        // [IDA] std::string SSO: spell name (CastInfo+0x20)
    constexpr auto MissileName      = 0x308;        // [IDA] std::string SSO: missile name (CastInfo+0x48)
    constexpr auto StartPos         = 0x388;        // [IDA] Vec3: start position (CastInfo+0xC8)
    constexpr auto EndPos           = 0x394;        // [IDA] Vec3: end position (CastInfo+0xD4)
    constexpr auto CastEndPos       = 0x3A4;        // [IDA] Vec3: cast end position (CastInfo+0xE4)
    constexpr auto CasterNetId      = 0x358;        // [IDA] int: source caster net id (CastInfo+0x98)
    constexpr auto TargetNetId      = 0x35C;        // [IDA] int: target net id (CastInfo+0x9C)
    constexpr auto CI_TargetNetId2  = 0x360;        // [IDA] int: secondary target (CastInfo+0xA0)
    constexpr auto CI_MissileNetId  = 0x364;        // [IDA] int: missile net id (CastInfo+0xA4)
 
    // --- CastInfo relative offsets (for code that needs CI base + offset pattern) ---
    constexpr auto CI_REL_SpellData    = 0x00;      // [IDA] CastInfo+0x00
    constexpr auto CI_REL_SpellName    = 0x20;      // [IDA] CastInfo+0x20
    constexpr auto CI_REL_MissileName  = 0x48;      // [IDA] CastInfo+0x48
    constexpr auto CI_REL_StartPos     = 0xC8;      // [IDA] CastInfo+0xC8
    constexpr auto CI_REL_EndPos       = 0xD4;      // [IDA] CastInfo+0xD4
    constexpr auto CI_REL_CastEndPos   = 0xE4;      // [IDA] CastInfo+0xE4
    constexpr auto CI_REL_CasterNetId  = 0x98;      // [IDA] CastInfo+0x98
    constexpr auto CI_REL_MissileNetId = 0xA4;      // [IDA] CastInfo+0xA4
 
    // --- Legacy aliases ---
    constexpr auto NetworkId        = MissileNetId; // 0x364
    constexpr auto SpellDataInst    = CI_SpellData; // 0x2C0
}

// ================================================================
// BASIC ATTACK / MISC
// ================================================================
namespace BasicAttack {
    constexpr auto Base             = 0x2C68;       // [D]
    constexpr auto Offset1          = 0x2C0;        // [D]
    constexpr auto Offset2          = 0x70;         // [D]
}

namespace Minion {
    constexpr auto LaneArray        = 0x68;         // [D] ptr to lane minion array (relative to MinionManager)
    constexpr auto LaneCount        = 0x70;         // [IDA] count of lane minions (relative to MinionManager)
    constexpr auto LaneType         = 0x4CC9;       // [CE] byte on obj: 4=Melee, 5=Ranged, 6=Cannon, 7=Super
}

// ================================================================
// DRAGON — Offsets for dragon soul detection (IDA sub_456A90 + sub_457DE0)
// ================================================================
namespace Dragon {
    constexpr auto CharacterHash    = 0x68;          // [IDA] DWORD hash on CharacterData (obj+CharData → +0x68)
    // Dragon Name Hash Table (global dword_1D995C0, 9 entries × 40 bytes)
    constexpr auto HashTable        = 0x1D995C0;     // [IDA] static hash table base
    constexpr auto HashTableEnd     = 0x1D99728;     // [IDA] end sentinel
    constexpr auto HashEntrySize    = 0x28;          // 40 bytes per entry (10 DWORDs)
    // Pre-computed dragon name hashes (sub_1074EA0 on dragon names)
    constexpr auto HashAir          = 0x11D34E07;    // SRU_Dragon_Air     → Cloud
    constexpr auto HashFire         = 0x99A9F7D9;    // SRU_Dragon_Fire    → Infernal
    constexpr auto HashWater        = 0x27F69DF4;    // SRU_Dragon_Water   → Ocean
    constexpr auto HashEarth        = 0x606D3187;    // SRU_Dragon_Earth   → Mountain
    constexpr auto HashHextech      = 0xA0808ACE;    // SRU_Dragon_Hextech → Hextech
    constexpr auto HashChemtech     = 0xF94EBA26;    // SRU_Dragon_Chemtech→ Chemtech
    constexpr auto HashRuined       = 0x518A146A;    // SRU_Dragon_Ruined  → Ruined
    constexpr auto HashElder        = 0x5944DC07;    // SRU_Dragon_Elder   → Elder
    constexpr auto HashParty        = 0x4B962AA3;    // SRU_Dragon_Party   → Party
}

// ================================================================
// SPELL CAST INFO (Active Spell)
// From: OnProcessSpell (0x920430) decompilation + chimera
// ================================================================
namespace SpellCastInfo {
    constexpr auto SpellData        = 0x0;          // [IDA] first QWORD = SpellData ptr
    constexpr auto SrcIndex         = 0x98;         // [C] source caster network index
    constexpr auto StartPos         = 0xD8;         // [C] Vec3 spell start position
    constexpr auto EndPos           = 0xE4;         // [C] Vec3 spell end position
    constexpr auto CastPos          = 0xF0;         // [C] Vec3 cast position
    constexpr auto TargetIndex      = 0x108;        // [C] target network index
    constexpr auto CastDelay        = 0x118;        // [C] float cast delay
    constexpr auto IsSpell          = 0x134;        // [C] bool is spell (not auto)
    constexpr auto IsSpecialAttack  = 0x13E;        // [C] bool is special attack
    constexpr auto IsAuto           = 0x141;        // [IDA] byte: is auto attack (chimera=0x13F)
    constexpr auto Slot             = 0x14C;        // [IDA] DWORD: spell slot index (chimera=0x148)
}

// ================================================================
// ITEM SYSTEM
// From: IDA MCP analysis + chimera_structures.h
// ================================================================
namespace ItemSystem {
    // GameObject::ItemList = 0x4D20 (in GameObject namespace)
    // Array of 7 ItemSlot pointers (6 items + trinket)
    constexpr auto SlotInfo         = 0x10;         // [IDA] ItemSlot+0x10 → ItemInfo ptr
    constexpr auto InfoData         = 0x38;         // [IDA] ItemInfo+0x38 → ItemData ptr
    constexpr auto InfoStacks       = 0x64;         // [C] ItemInfo+0x64 → stack count
    constexpr auto DataItemId       = 0xB4;         // [IDA] ItemData+0xB4 → item ID int
    constexpr auto DataAbilityHaste = 0x160;        // [C] ItemData stat
    constexpr auto DataHealth       = 0x164;        // [C] ItemData stat
    constexpr auto DataArmor        = 0x19C;        // [C] ItemData stat
    constexpr auto DataMR           = 0x1BC;        // [C] ItemData stat
    constexpr auto DataAD           = 0x1D8;        // [C] ItemData stat
    constexpr auto DataAP           = 0x1E0;        // [C] ItemData stat
    constexpr auto DataAtkSpeedMult = 0x20C;        // [C] ItemData stat
}

// ================================================================
// NAV GRID
// Source: sig 48 8B 05 ? ? ? ? 0F 28 DA → Global::NavGrid (0x1D7DD08)
// Chain: navGridPtr → +0x8 → NavGridManager → fields below
// IDA MCP verified (2026-03-11): decompile of GetCollisionFlags
// (0x1195B80), sub_1195BC0, sub_1190840, sub_119C040, sub_119C380,
// sub_119C210, sub_119C4F0 — all access *(qword_1D7DD08 + 8) = mgr
//
// KEY FIX: MinX/MinZ were WRONG (0x30/0x38).
// Decompile shows mgr[59] and mgr[61] → float at 59*4=0xEC, 61*4=0xF4
// This was causing intermittent bush/wall detection failure.
// ================================================================
namespace NavGrid {
    // Pointer chain
    constexpr auto NavGridMgr       = 0x8;          // [IDA] navGridPtr → +0x8 → manager

    // Map bounds (float)
    constexpr auto MinX             = 0xEC;         // [IDA] mgr[59] = world min X coordinate
    constexpr auto MinZ             = 0xF4;         // [IDA] mgr[61] = world min Z coordinate
    constexpr auto MaxX             = 0xF8;         // [IDA] mgr[62] = world max X coordinate
    constexpr auto MaxZ             = 0x100;        // [IDA] mgr[64] = world max Z coordinate
 
    // Cell data
    constexpr auto Data             = 0x110;        // [IDA] mgr+272 = ptr to cell array (16 bytes per cell)
    constexpr auto Width            = 0x708;        // [IDA] mgr+1800 = grid width (cells)
    constexpr auto Height           = 0x70C;        // [IDA] mgr+1804 = grid height (cells)
 
    // Scale
    constexpr auto InverseScale     = 0x714;        // [IDA] mgr+1812 = 1/cellSize (MULTIPLY to get cell index)
    constexpr auto Scale            = 0x710;        // [IDA] mgr[452] = cell size (used in bounds check)
 
    // Grass/Brush detection
    constexpr auto GrassRegions     = 0x158;        // [IDA] mgr+344 = grass region bitfield ptr
 
    // Cell structure: 16 bytes per cell
    // Layout: [uint64_t ptrData][uint16_t flags][uint16_t pad][uint32_t pad]
    // If ptrData != 0: real flags = *(uint16_t*)(ptrData + 6)
    // If ptrData == 0: real flags = cell.flags (at cell + 8)
    constexpr auto CellSize         = 16;           // [IDA] bytes per cell
 
    // Collision flag bitmask (from decompile of multiple functions)
    constexpr uint16_t FLAG_WALL    = 0x0001;       // [IDA] sub_119C380: bit 0 = wall
    constexpr uint16_t FLAG_NOWALK  = 0x0002;       // [IDA] sub_119C210: bit 1 = not walkable
    constexpr uint16_t FLAG_BRUSH   = 0x0C00;       // [IDA] sub_119C140: bits 10-11 = brush/grass
    constexpr uint16_t FLAG_SPECIAL = 0x1000;       // [IDA] sub_119C040: bit 12 = special terrain
}

// ================================================================
// MANAGER LIST
// ================================================================
namespace ManagerList {
    constexpr auto Items            = 0x8;          // [S]
    constexpr auto Size             = 0x10;         // [S]
}

// ================================================================
// MINIMAP
// ================================================================
namespace Minimap {
    constexpr auto MinimapParent    = 0x1D7A3D0;    // [CE] global ptr (same as NetInstance)
    constexpr auto MinimapHud       = 0x3B8;         // [CE] MinimapParent->+0x3B8 (was 0x288 in 14.23)
    constexpr auto HudVisible       = 0xD8;          // [CE] MinimapHud+0xD8 byte flag
}

// ================================================================
// EXTRA GLOBALS
// ================================================================
namespace Extra {
    constexpr auto TurretManager    = 0x1D87068;    // [P][IDA] 20 xrefs confirmed
    constexpr auto ViewMatrixInst   = 0x1E2C070;    // [P] view/projection matrix (from offsetplugin.hpp)
    constexpr auto IsClone          = 0x2BB2B0;     // [P] function RVA (+0x10)
}

// ================================================================
// VTABLES
// ================================================================
namespace VTable {
    constexpr auto AIMinionClient   = 0x18DD7F0;    // [P]
}

// ================================================================
// JUNGLE MONSTER NAME STRINGS
// These are string addresses in the binary - version specific!
// Found via IDA MCP find_regex on binary 0x2342000
// NOTE: These are for the IDA binary, NOT the dump binary!
//       For dump binary (0x202D000), re-scan needed.
// ================================================================
namespace JungleNames {
    // IDA binary (0x2342000) string addresses:
    constexpr auto SRU_RiftHerald   = 0x18d5358;    // [IDA] "SRU_RiftHerald"
    constexpr auto SRU_Horde        = 0x18d6690;    // [IDA] "SRU_Horde"
    constexpr auto SRU_Dragon       = 0x18d66B0;    // [IDA] "SRU_Dragon"
    constexpr auto SRU_Dragon_Elder = 0x18d66C0;    // [IDA] "SRU_Dragon_Elder"
    constexpr auto SRU_Baron        = 0x18e58D0;    // [IDA] "SRU_Baron"
}

// ================================================================
// OBJECT TYPE FLAGS (obfuscated field at obj+0x4C)
// Checked via CompareTypeFlags (sub_29CD30) — do NOT read directly!
// Use: Function::CompareTypeFlags(obj, FLAG_xxx)
// Found via IDA MCP decompile of sub_3088A0, sub_308B50, sub_3089A0, sub_308C70
// ================================================================
namespace TypeFlags {
    constexpr auto ObfuscatedField  = 0x4C;          // [IDA] obj+76 in sub_29CD30
    // Bit flags passed to CompareTypeFlags:
    constexpr auto Minion           = 0x0400;         // [IDA] sub_3089A0: flag 1024
    constexpr auto Hero             = 0x0800;         // [IDA] sub_308B50: flag 2048
    constexpr auto JungleMonster    = 0x2000;         // [IDA] sub_3088A0: flag 8192 (IsJungleMonster)
    constexpr auto LargeMonster     = 0x0080;         // [IDA] sub_345650: "Monster_Large" flag
    constexpr auto BuffMonster      = 0x0100;         // [IDA] sub_345650: "Monster_Buff" flag
    constexpr auto MinionSummon     = 0x0100;         // [IDA] sub_345650: "Minion_Summon" flag (same bit)
    constexpr auto Plant            = 0x8000;         // [IDA] sub_345650: "Plant" flag 32768
    constexpr auto CampMonster      = 0x10000;        // [IDA] sub_345650: 0x10000 after Plant
    constexpr auto Crab             = 0x2000;         // [IDA] sub_345650: "Monster_Crab" flag
    constexpr auto IsFleeing        = 0x0200;         // [IDA] sub_345650: fleeing check flag
    constexpr auto AttackableObj    = 0x0008;         // [IDA] sub_345650: attackable
    constexpr auto VisibleObj       = 0x0010;         // [IDA] sub_345650: visible flag
    constexpr auto RenderTarget     = 0x0020;         // [IDA] sub_345650: render target
    constexpr auto IsRecalling      = 0x4000;         // [IDA] sub_345650: recall check
    constexpr auto HasUltimate      = 0x20000;        // [IDA] sub_345650: vtable+2552 check
}

// ================================================================
// MINION CLASSIFICATION (from sub_BBB10 RegisterProperty table)
// LaneMinionType byte value on the minion object, registered via
// sub_10D1B80 with string name + numeric class ID
// Access: use GetJungleType (Function::GetJungleType) or read
//         the byte at the correct offset after finding it at runtime
// Found via IDA MCP decompile of sub_BBB10
// ================================================================
namespace MinionClass {
    // Class IDs (byte values):
    constexpr auto Unset            = 0;              // [IDA] v50=0 "Unset"
    constexpr auto Pet              = 1;              // [IDA] v54=1 "Pet"
    constexpr auto JungleMonster    = 2;              // [IDA] v58=2 "JungleMonster"
    constexpr auto TeamMinion       = 3;              // [IDA] v62=3 "TeamMinion"
    constexpr auto MeleeLaneMinion  = 4;              // [IDA] v66=4 "MeleeLaneMinion"
    constexpr auto RangedLaneMinion = 5;              // [IDA] v70=5 "RangedLaneMinion"
    constexpr auto SiegeLaneMinion  = 6;              // [IDA] v74=6 "SiegeLaneMinion"
    constexpr auto SuperLaneMinion  = 7;              // [IDA] v78=7 "SuperLaneMinion"
}

// ================================================================
// JUNGLE TYPE (from CharacterData sub-object)
// sub_345410 returns *(uint32_t*)(charData + 0x4A84)
// charData = obj + GameObject::CharacterData (0x40C8)
// GetJungleType (sub_66CE60) maps these to:
//   1 → type:0 (Normal),  2 → type:2 (Buff/Dragon), 3 → type:1 (Baron-like)
// Found via IDA MCP decompile of sub_345410 (returns charData+19076)
// ================================================================
namespace JungleType {
    constexpr auto TypeOffset       = 0x4A84;         // [IDA] charData + 19076 in sub_345410

    // Return values from GetJungleType:
    constexpr auto Normal           = 0;              // [IDA] sub_66CE60: case v23-1
    constexpr auto Baron            = 1;              // [IDA] sub_66CE60: v24==0 → return 1
    constexpr auto Dragon           = 2;              // [IDA] sub_66CE60: v22==0 → return 2
}

// ================================================================
// PLANT IDENTIFICATION
// Plants are identified via TypeFlags::Plant (0x8000)
// checked through CompareTypeFlags function
// Plant string names (IDA):
//   "Plant"             @ 0x18EF538
//   "OnPlantActivated"  @ 0x1902660
//   "AttackVisionplant" @ 0x18EBDA0
// Dragon subtypes (IDA string addresses):
//   SRU_Dragon_Air      @ 0x1908F78
//   SRU_Dragon_Fire     @ 0x1908F88
//   SRU_Dragon_Water    @ 0x1908F98
//   SRU_Dragon_Earth    @ 0x1908FB0
//   SRU_Dragon_Ruined   @ 0x1908FC8
//   SRU_Dragon_Hextech  @ 0x1908FE8
//   SRU_Dragon_Chemtech @ 0x1909000
//   SRU_Dragon_Party    @ 0x1909018
// ================================================================
namespace PlantInfo {
    // Plants are checked via: CompareTypeFlags(obj, TypeFlags::Plant)
    // Plant types are distinguished by CharacterName (obj + 0x4330):
    //   "SRU_Plant_Health"   → Honeyfruit (healing plant)
    //   "SRU_Plant_Satchel"  → Blast Cone (knockback plant)
    //   "SRU_Plant_Vision"   → Scryer's Bloom (vision plant)
}

} // namespace Offset
wow thank you very much. i'll try the dumper again.
kral84 is offline

Old 15th March 2026, 08:10 AM   #12978
chen399516
n00bie

chen399516's Avatar

Join Date: Jan 2026
Posts: 17
Reputation: 10
Rep Power: 7
chen399516 has made posts that are generally average in quality
Points: 268, Level: 1
Points: 268, Level: 1 Points: 268, Level: 1 Points: 268, Level: 1
Level up: 67%, 132 Points needed
Level up: 67% Level up: 67% Level up: 67%
Activity: 9.4%
Activity: 9.4% Activity: 9.4% Activity: 9.4%
Quote:
Originally Posted by trankhanhtinh1 View Post
use version update for offset correct

The previous version was just a test, to find errors and make improvements. Now I can confirm that my LOLDUMPER dll can produce accurate results.

Code:
# pragma once
# include <cstdint>

// ================================================================
// League of Legends - Offsets
// Updated: 2026-03-12 (Hotfix) (LOLDumper v5.0 + offsetplugin.hpp + IDA MCP)
// Binary: League of Legends.exe
// Global/Function RVAs: from module size 0x202D000 (dump files)
// Struct offsets: verified via IDA on module size 0x2342000
// Base: 0x0 (relative offsets from module base)
//
// Sources:
//   [D]   = LOLDumper_full.h (pattern-scanned)
//   [P]   = offsetplugin.hpp (ida_lol_plugin.dll output)
//   [IDA] = IDA Pro MCP verified (decompile/disasm confirmed)
//   [CE]  = Cheat Engine verified at runtime
//   [S]   = struct offsets (unchanged between versions)
//   [C]   = chimera_structures.h reference (needs CE verify)
//
// Hotfix notes (2026-03-12):
//   - Function RVAs shifted +0x10 from 2026-03-05 hotfix
//   - All globals remained STABLE (confirmed via LOLDumper scan)
//   - Struct offsets STABLE (RegisterProperty-based, version-independent)
//   - LOLDumper re-scan confirmed globals unchanged
//   - Function deltas: IssueOrder +0x10, IsAlive +0x10, GetAttackDelay +0x10, etc.
//   - CastSpellSafe still at same RVA (offsetplugin.hpp: 0xBB9E60 → needs +0x10 verify)
//   - DetectionWatcher2 for Chimera-style mainloop_check is currently
//     resolved at runtime by signature: 4C 8B 3D ? ? ? ? 4D 85 FF 0F
//   - Current packet pipeline:
//       CastSpellSafe -> CastSpellPacketA/B/Charged -> PacketSendCommon -> PacketSerializeCommon
//       IssueOrderCore -> IssueOrderPacketBuilder -> PacketSendCommon -> PacketSerializeCommon
// ================================================================

namespace Offset {

// ================================================================
// GLOBAL POINTERS / INSTANCES  (all stable across hotfix)
// ================================================================
namespace Global {
    constexpr auto LocalPlayer      = 0x1DAB760;   // [D][P] local player ptr
    constexpr auto HeroManager      = 0x1D7A470;   // [D][P] hero list ptr
    constexpr auto GameTime         = 0x1D88580;   // [D][P] game time float
    constexpr auto MissileManager   = 0x1D7DD90;   // [D] missile manager ptr
    constexpr auto NavGrid          = 0x1D7DD08;   // [D] navigation grid ptr
    constexpr auto HudInstance      = 0x1D7A5B8;   // [D][P] HUD instance ptr
    constexpr auto UnderMouseObj    = 0x1D7DF90;   // [D] object under mouse cursor
    constexpr auto ViewPort         = 0x1D8D1F0;   // [D] viewport ptr
    constexpr auto ObjectManager    = 0x1D7A418;   // [D][P] object manager instance
    constexpr auto MinionManager    = 0x1D7A468;   // [IDA] minion+jungle list (CastSpellSafe decompile: qword_1D7A468)
    constexpr auto NetInstance      = 0x1D7A410;   // [IDA] net instance (Script-New had 0x1D7A3D0, new build +0x40)
    constexpr auto CursorInstance   = 0x1E056D8;   // [P] cursor position (Vec3)
    constexpr auto MouseScreenVec2  = 0x1D7DD38;   // [D] mouse 2D screen position
    constexpr auto ChatClient       = 0x1D8D240;   // [IDA] fallback, needs verification
    constexpr auto ChatInstance     = 0x1D7DFA0;   // [IDA] fallback
    constexpr auto r3dRenderer      = 0x1E3FEB8;   // [D] renderer instance (oViewPort2)
    constexpr auto ViewPort2        = 0x1E3FEB8;   // [D] viewport2/renderer
    constexpr auto MySpellState     = 0x1D80AE0;   // [D] spell state global
    constexpr auto TurretManager    = 0x1D870A8;   // [P] turret list
    constexpr auto ShopInstance     = 0x1D8D258;   // [IDA] fallback
    constexpr auto OpenWindowsArray = 0x1E3DC58;   // [IDA] fallback
    constexpr auto OpenWindowsCount = 0x1E3DC60;   // [IDA] fallback
}

// ================================================================
// FLAGS  (stable across hotfix - confirmed via decompile)
// ================================================================
namespace Flag {
    constexpr auto IssueOrderFlag   = 0x1CDDF88;   // [D][IDA] dword_1CDDF88 in IssueOrder (Chimera: order + 17)
    constexpr auto IssueOrder       = IssueOrderFlag; // Backward-compatible alias
    constexpr auto CastSpellFlag    = 0x1CDDF20;   // [D][IDA] byte_1CDDF20 in CastSpellSafe (Chimera CastSpellFlag)
    constexpr auto CastSpell        = CastSpellFlag; // Backward-compatible alias
}

// ================================================================
// FUNCTIONS (RVAs) — UPDATED for hotfix 2026-03-05
// ================================================================
namespace Function {
    // Core — 2026-03-12 hotfix (+0x10 from 03-05)
    constexpr auto IssueOrderCore       = 0x29FC20;     // [D] was 0x29FC10
    constexpr auto IssueOrder           = IssueOrderCore;
    constexpr auto IssueOrderPacketBuilder = 0x360CB0;  // Fallback (+0x10)
    constexpr auto IssueOrderPacketPostSend = 0x2CE8D0; // Fallback (+0x10)
    constexpr auto WorldToScreen        = 0x1241600;    // [D] was 0x1241370
    constexpr auto CastSpellWrapper     = 0x1E9A80;     // [D] was 0x1E9A70
    constexpr auto CastSpellSafe        = 0xBB9E00;     // [IDA] sub_BB9E00 (was 0xBB9E70 = MIDDLE of func!)
    constexpr auto CastSpellPacketA     = 0x91BF10;     // Fallback (+0x10)
    constexpr auto CastSpellPacketB     = 0x91B6C0;     // Fallback (+0x10)
    constexpr auto CastSpellPacketCharged = 0x91C7D0;   // Fallback (+0x10)
    constexpr auto PacketSendCommon     = 0x686940;     // Fallback (+0x10)
    constexpr auto PacketSerializeCommon = 0x686980;    // Fallback (+0x10)
    constexpr auto PrintChat            = 0x1095120;    // [P] (+0x10)
    constexpr auto GetBoundingRadius    = 0x285650;     // [D] was 0x285640
    constexpr auto GetAttackDelay       = 0x52C5A0;     // [D] was 0x52C590
    constexpr auto GetAttackWindup      = 0x52C4A0;     // [D] was 0x52C490
    constexpr auto GetCollisionFlags    = 0x1195E10;    // [D] was 0x1195B80
    constexpr auto GetPing              = 0x669EB0;     // [D] was 0x669F10

    // Object Iteration
    constexpr auto GetFirstObject       = 0x512920;     // [D] was 0x512910
    constexpr auto GetFirstObjectAlt    = 0x9D0410;     // [P] (+0x10)
    constexpr auto GetNextObject        = 0x513410;     // [D] was 0x513400
    constexpr auto FindObject           = 0x512110;     // [P] (+0x10)
    constexpr auto GetAiManager         = 0x50AAE0;     // [D] was 0x50AAD0
    constexpr auto GetAIManagerAlt      = 0x28D410;     // [P] (+0x10)
 
    // Type Checks
    constexpr auto IsTurret             = 0x308600;     // [D] was 0x3085F0
    constexpr auto IsHero               = 0x308700;     // [D] was 0x3086F0
    constexpr auto IsBuilding           = 0x308830;     // [P] (+0x10)
    constexpr auto IsAlive              = 0x2E6360;     // [D] was 0x2E6350
    constexpr auto IsDead               = 0x29B390;     // [P] (+0x10)
    constexpr auto IsTargetableByUnit   = 0x29E290;     // [P] (+0x10)
    constexpr auto IsVulnerable         = 0x29C050;     // [P] (+0x10)
    constexpr auto IsJungleMonster      = 0x29C220;     // [P] (+0x10)
    constexpr auto IsDragon             = 0x29B640;     // [P] (+0x10)
    constexpr auto IsElderDragon        = 0x29B6B0;     // [P] (+0x10)
    constexpr auto IsBaron              = 0x29AAA0;     // [P] (+0x10)
    constexpr auto IsSelectable         = 0x212180;     // [P] (+0x10)
    constexpr auto CompareTypeFlags     = 0x29CD40;     // [P] (+0x10)
    constexpr auto IsFleeing            = 0x20F340;     // [P] (+0x10)
    constexpr auto IsNoRender           = 0x20F390;     // [P] (+0x10)
    constexpr auto GetJungleType        = 0x66CE70;     // [P] (+0x10)
 
    // Attack / Combat
    constexpr auto CanAttack            = 0x1F90E0;     // [P] (+0x10)
    constexpr auto GetSpellCastInfo     = 0x283F20;     // [P] (+0x10)
    constexpr auto GetSpellSlot         = 0x90AA50;     // [P] (+0x10)
    constexpr auto GetResourceType      = 0x281240;     // [P] (+0x10)
    constexpr auto HasBuffOfType        = 0x296410;     // [P] (+0x10)
    constexpr auto GetGoldRedirectTgt   = 0x1FF9A0;     // [P] (+0x10)
 
    // Level Up
    constexpr auto LevelSpell           = 0xBA39C0;     // Fallback (+0x10)
 
    // Map / Minimap
    constexpr auto GetMapID             = 0x28E320;     // [D] was 0x28E310
 
    // Hooks / Callbacks
    constexpr auto OnCreateObject       = 0x517E20;     // [P] (+0x10)
    constexpr auto OnGameUpdate         = 0x5111C0;     // [P] (+0x10)
    constexpr auto OnProcessSpell       = 0x920590;     // [P] (+0x10)
    constexpr auto OnSpellImpact        = 0x917CA0;     // [P] (+0x10)
    constexpr auto OnStopCast           = 0x9208A0;     // [P] (+0x10)
    constexpr auto OnFinishCast         = 0x2C5770;     // [P] (+0x10)
    constexpr auto OnBuffAdd            = 0xBCDE90;     // [P] (+0x10)
    constexpr auto CreateClientEffect   = 0x869E90;     // [P] (+0x10)
}

// ================================================================
// GAME OBJECT STRUCT  (stable - struct offsets don't change)
// ================================================================
namespace GameObject {
    constexpr auto Index            = 0x10;         // [S]
    constexpr auto Team             = 0x3C;         // [S]
    constexpr auto Name             = 0x58;         // [S]
    constexpr auto NetId            = 0xCC;         // [D][S]
    constexpr auto Dead             = 0x250;        // [S]
    constexpr auto TeamAlt          = 0x259;        // [D]
    constexpr auto Position         = 0x25C;        // [S]
    constexpr auto EffectEmitter    = 0x258;        // [S]
    constexpr auto Visibility       = 0x2E0;        // [S]
    constexpr auto MissileClient    = 0x2D8;        // [S]
    constexpr auto Visible          = 0x308;        // [CE] verified: 0=fog, 1=visible on screen
    constexpr auto IsInvulnerable   = 0x5A0;        // [S]
    constexpr auto Radius           = 0x6F8;        // [D]
    constexpr auto RecallState      = 0xF48;        // [S]
    constexpr auto CharacterName    = 0x4330;       // [D]
    constexpr auto CharacterData    = 0x40C8;       // [S]
    constexpr auto Direction        = 0x21D8;       // [C] facing direction Vec3 (FaceDirection_s)
    constexpr auto ItemList         = 0x4D20;       // [C] array of 7 ItemSlot ptrs (6 items + trinket)
}

// ================================================================
// MANA
// ================================================================
namespace Mana {
    constexpr auto MP               = 0x360;        // [S]
    constexpr auto MaxMP            = 0x388;        // [S]
}

// ================================================================
// HEALTH (LeagueObfuscation<float>, 0x28 apart)
// ================================================================
namespace Health {
    constexpr auto HP               = 0x1080;       // [D]
    constexpr auto MaxHP            = 0x10A8;       // [D]
    constexpr auto HPMaxPenalty     = 0x10D0;       // [D]
    constexpr auto AllShield        = 0x1120;       // [D]
    constexpr auto PhysicalShield   = 0x1148;       // [D]
    constexpr auto MagicalShield    = 0x1170;       // [D]
    constexpr auto ChampSpecific    = 0x1198;       // [D]
    constexpr auto InHealAllied     = 0x11C0;       // [IDA] sub_2E3220: HP+320=0x1080+0x140
    constexpr auto InHealEnemy      = 0x11E8;       // [IDA] sub_2E3220: HP+360=0x1080+0x168
    constexpr auto InDamage         = 0x1210;       // [IDA] sub_2E3220: HP+400=0x1080+0x190
    constexpr auto StopShieldFade   = 0x1238;       // [IDA] sub_2E3220: HP+440=0x1080+0x1B8
}

// ================================================================
// TARGETABLE
// ================================================================
namespace Targetable {
    constexpr auto IsTargetable     = 0xED0;        // [D]
    constexpr auto TargetableFlags  = 0xEF8;        // [IDA] mIsTargetableToTeamFlags string xref
}

// ================================================================
// ACTION STATE
// ================================================================
namespace ActionState {
    constexpr auto State1           = 0x1470;       // [IDA] lea rdx,[rsi+1470h] -> sub_1FD490 "ActionState"
    constexpr auto State2           = 0x14A8;       // [IDA] 0x1470+0x38 -> sub_1FD490 "ActionState2"
}

// ================================================================
// DAMAGE MODIFIERS
// ================================================================
namespace DamageModifier {
    constexpr auto PhysDmgPercent   = 0x0E78;       // [IDA] lea rcx,[r14+0E78h] "mPhysicalDamagePercentageModifier"
    constexpr auto MagicDmgPercent  = 0x0EA0;       // [IDA] lea rcx,[r14+0EA0h] "mMagicalDamagePercentageModifier"
}

// ================================================================
// HERO STATS (LeagueObfuscation<float>, 0x28 apart)
// Stat block base: obj + 0x1B88
// ================================================================
namespace HeroStats {
    constexpr auto Base                     = 0x1B88;       // [D]

    // Cooldown / Ability Haste
    constexpr auto PercentCooldownMod       = 0x1B88;       // [D] base + 0x0
    constexpr auto AbilityHaste             = 0x1BB0;       // [D] base + 0x28
    constexpr auto PercentCooldownCapMod    = 0x1BD8;       // [D] base + 0x50
    constexpr auto PassiveCdEndTime         = 0x1C00;       // [D] base + 0x78
    constexpr auto PassiveCdTotalTime       = 0x1C28;       // [D] base + 0xA0
 
    // Minion-specific
    constexpr auto PercentDmgToBarracksMin  = 0x1C50;       // [D] base + 0xC8
    constexpr auto FlatDmgReducBarracks     = 0x1C78;       // [D] base + 0xF0
    constexpr auto IncreasedMoveSpeedMinion = 0x1CA0;       // [D] base + 0x118
 
    // Physical Damage
    constexpr auto FlatPhysicalDmgMod       = 0x1CC8;       // [D] base + 0x140
    constexpr auto PercentPhysicalDmgMod    = 0x1CF0;       // [D] base + 0x168
    constexpr auto PercentBonusPhysDmgMod   = 0x1D18;       // [D] base + 0x190
    constexpr auto PercentBasePhysDmgFlat   = 0x1D40;       // [D] base + 0x1B8
 
    // Magic Damage
    constexpr auto FlatMagicDmgMod          = 0x1D68;       // [D] base + 0x1E0
    constexpr auto PercentMagicDmgMod       = 0x1D90;       // [D] base + 0x208
    constexpr auto FlatMagicReduction       = 0x1DB8;       // [D] base + 0x230
    constexpr auto PercentMagicReduction    = 0x1DE0;       // [D] base + 0x258
 
    // Cast Range
    constexpr auto FlatCastRangeMod         = 0x1E08;       // [D] base + 0x280
 
    // Attack Speed
    constexpr auto AttackSpeedMod           = 0x1E30;       // [D] base + 0x2A8
    constexpr auto PercentAttackSpeedMod    = 0x1E58;       // [D] base + 0x2D0
    constexpr auto PercentMultiAtkSpeedMod  = 0x1E80;       // [D] base + 0x2F8
 
    // Healing
    constexpr auto PercentHealingAmountMod  = 0x1EA8;       // [D] base + 0x320
 
    // Attack Damage
    constexpr auto BaseAttackDamage         = 0x1ED0;       // [D] base + 0x348
    constexpr auto BaseAtkDmgSansScale      = 0x1EF8;       // [D] base + 0x370
    constexpr auto FlatBaseAtkDmgMod        = 0x1F20;       // [D] base + 0x398
    constexpr auto PercentBaseAtkDmgMod     = 0x1F48;       // [D] base + 0x3C0
 
    // Ability Power
    constexpr auto BaseAbilityDamage        = 0x1F70;       // [D] base + 0x3E8
 
    // Crit
    constexpr auto CritDamageMultiplier     = 0x1F98;       // [D] base + 0x410
    constexpr auto ScaleSkinCoef            = 0x1FC0;       // [D] base + 0x438
    constexpr auto Dodge                    = 0x1FE8;       // [D] base + 0x460
    constexpr auto Crit                     = 0x2010;       // [D] base + 0x488
 
    // Base HP Pool
    constexpr auto FlatBaseHPPoolMod        = 0x2038;       // [D] base + 0x4B0
 
    // Armor & MR
    // NOTE: Armor (0x2060) is TOTAL armor — already includes base + bonus.
    //       BonusArmor removed intentionally; use Armor directly for all calcs.
    constexpr auto Armor                    = 0x2060;       // [D] base + 0x4D8  (TOTAL armor — use this)
    // BonusArmor                           = 0x2088        // REMOVED — would double-count vs. Armor total
    constexpr auto SpellBlock               = 0x20B0;       // [D] base + 0x528  (MR, total)
    constexpr auto BonusSpellBlock          = 0x20D8;       // [D] base + 0x550
 
    // HP Regen
    constexpr auto HPRegenRate              = 0x2100;       // [D] base + 0x578
    constexpr auto BaseHPRegenRate          = 0x2128;       // [D] base + 0x5A0
 
    // Movement
    constexpr auto MoveSpeed                = 0x2150;       // [D] base + 0x5C8
    constexpr auto MoveSpeedBaseIncrease    = 0x2178;       // [D] base + 0x5F0
    constexpr auto AttackRange              = 0x21A0;       // [D] base + 0x618
 
    // Bubble Radius
    constexpr auto FlatBubbleRadiusMod      = 0x21C8;       // [D] base + 0x640
    constexpr auto PercentBubbleRadiusMod   = 0x21F0;       // [D] base + 0x668
 
    // Armor Penetration
    constexpr auto FlatArmorPen             = 0x2218;       // [D] base + 0x690
    constexpr auto PhysicalLethality        = 0x2240;       // [D] base + 0x6B8
    constexpr auto PercentArmorPen          = 0x2268;       // [D] base + 0x6E0
    constexpr auto PercentBonusArmorPen     = 0x2290;       // [D] base + 0x708
    constexpr auto PercentCritBonusArmorPen = 0x22B8;       // [D] base + 0x730
    constexpr auto PercentCritTotalArmorPen = 0x22E0;       // [D] base + 0x758
 
    // Magic Penetration
    constexpr auto FlatMagicPen             = 0x2308;       // [D] base + 0x780
    constexpr auto MagicLethality           = 0x2330;       // [D] base + 0x7A8
    constexpr auto PercentMagicPen          = 0x2358;       // [D] base + 0x7D0
    constexpr auto PercentBonusMagicPen     = 0x2380;       // [D] base + 0x7F8
 
    // Lifesteal / Vamp
    constexpr auto PercentLifeSteal         = 0x23A8;       // [D] base + 0x820
    constexpr auto PercentSpellVamp         = 0x23D0;       // [D] base + 0x848
    constexpr auto PercentOmnivamp          = 0x23F8;       // [D] base + 0x870
    constexpr auto PercentPhysicalVamp      = 0x2420;       // [D] base + 0x898
 
    // Pathing
    constexpr auto PathfindingRadiusMod     = 0x2448;       // [D] base + 0x8C0
 
    // Misc
    constexpr auto PercentCCReduction       = 0x2470;       // [D] base + 0x8E8
    constexpr auto PercentEXPBonus          = 0x2498;       // [D] base + 0x910
 
    // Base Armor/MR Flat Mods
    constexpr auto FlatBaseArmorMod         = 0x24C0;       // [D] base + 0x938
    constexpr auto FlatBaseSpellBlockMod    = 0x24E8;       // [D] base + 0x960
 
    // Resource Regen
    constexpr auto PARRegenRate             = 0x2510;       // [D] base + 0x988
    constexpr auto PrimaryARBaseRegenRate   = 0x2538;       // [D] base + 0x9B0
    constexpr auto SecondaryARRegenRate     = 0x2560;       // [D] base + 0x9D8
    constexpr auto SecondaryARBaseRegenRate = 0x2588;       // [D] base + 0xA00
 
    // Base Attack Speed
    constexpr auto FlatBaseAttackSpeedMod   = 0x25B0;       // [D] base + 0xA28
}

// ================================================================
// HERO-SPECIFIC
// ================================================================
namespace Hero {
    constexpr auto Gold                 = 0x2830;   // [D]
    constexpr auto GoldTotal            = 0x2858;   // [D]
    constexpr auto MinimumGold          = 0x2880;   // [D]
    constexpr auto FollowerTargetDelay  = 0x2DB8;   // [D] minion follower delay
    constexpr auto CombatType           = 0x2C98;   // [IDA] lea rdi,[r14+2C98h] "mCombatType"
    constexpr auto Exp                  = 0x4CF0;   // [D]
    constexpr auto LevelRef             = 0x4D18;   // [IDA] lea rcx,[r14+4D18h] "mLevelRef"
    constexpr auto LevelUpPoints        = 0x4D78;   // [chimera] LevelRef + 0x60 = skill points available
    constexpr auto VisionScore          = 0x55E0;   // [D]
    constexpr auto ShutdownValue        = 0x5608;   // [D]
    constexpr auto BaseGoldOnDeath      = 0x5630;   // [D]
    constexpr auto NeutralMinionsKilled = 0x5658;   // [IDA] lea rcx,[r14+5658h] "mNumNeutralMinionsKilled"
}

// ================================================================
// LIFETIME PROPS
// ================================================================
namespace Lifetime {
    constexpr auto Lifetime         = 0x0DB0;       // [IDA] lea rcx,[r14+0DB0h] "mLifetime"
    constexpr auto MaxLifetime      = 0x0DD8;       // [IDA] lea rcx,[r14+0DD8h] "mMaxLifetime"
    constexpr auto LifetimeTicks    = 0x0E00;       // [IDA] lea rcx,[r14+0E00h] "mLifetimeTicks"
}

// ================================================================
// SPELLBOOK & SPELL SLOTS
// ================================================================
namespace SpellBook {
    constexpr auto Offset           = 0x30E8;       // [D]
    constexpr auto SpellSlotArray   = 0xAE0;        // [D]
    constexpr auto ActiveSpellCast  = 0x3120;       // SpellBook::Offset + 0x38

    // SpellSlot (SpellDataInst)
    constexpr auto SlotLevel        = 0x28;         // [S]
    constexpr auto SlotCooldown     = 0x30;         // [S]
    constexpr auto SlotStacks       = 0x5C;         // [S]
    constexpr auto SlotTotalCd      = 0x74;         // [S]
    constexpr auto SlotSpellInput   = 0x120;        // [IDA] SpellInput/TargetClient (LOLDumper scans 0xB8 - wrong)
    constexpr auto SlotSpellInfo    = 0x128;        // [IDA] SpellInfo ptr (LOLDumper scans 0xC0 - wrong)
 
    // SpellInput
    constexpr auto InputTargetNetId = 0x14;         // [S]
    constexpr auto InputStartPos    = 0x18;         // [S]
    constexpr auto InputEndPos      = 0x24;         // [S]
 
    // SpellInfo
    constexpr auto InfoSpellData    = 0x60;         // [S]
 
    // SpellData
    constexpr auto DataSpellName    = 0x80;         // [S]
    constexpr auto SpellInfoNamePtr = 0x28;         // [brute confirmed] ptr -> char*
    constexpr auto DataManaCost     = 0x5F4;        // [S]
    constexpr auto DataResource     = 0x8;          // [D]
 
    // SpellData → SpellDataResource (SpellData + 0x60)
    constexpr auto DataResourceBase = 0x60;         // [IDA] SpellData+0x60 → SpellDataResource ptr
    constexpr auto ResCastRange     = 0x478;        // [C] array of 7 floats (per rank)
    constexpr auto ResMissileSpeed  = 0x518;        // [C] float missile speed
    constexpr auto ResLineWidth     = 0x568;        // [C] float line width
    constexpr auto ResMaxAmmo       = 0x3C0;        // [C] array of 7 ints (per rank)
    constexpr auto ResCastType      = 0x510;        // [C] targeting type enum
    constexpr auto ResMissileSpec   = 0x508;        // [C] missile specification ptr
    constexpr auto ResScriptName    = 0x80;         // [C] spell script name string
    constexpr auto ResCooldownTime  = 0x304;        // [C] array of 7 floats (per rank)
    constexpr auto ResAmmoRecharge  = 0x408;        // [C] array of 7 floats
    constexpr auto ResImgIconName   = 0x2A0;        // [C] icon name string
}

// ================================================================
// BUFF MANAGER
// ================================================================
namespace BuffManager {
    constexpr auto Offset           = 0x28B8;       // [D]
    constexpr auto EntriesEnd       = 0x10;         // [S]
    constexpr auto EntryBuff        = 0x10;         // [S]
    constexpr auto BuffType         = 0x0C;         // [S]
    constexpr auto BuffNamePtr      = 0x10;         // [S]
    constexpr auto BuffNameStr      = 0x8;          // [S]
    constexpr auto BuffStartTime    = 0x18;         // [S]
    constexpr auto BuffEndTime      = 0x1C;         // [S]
    constexpr auto BuffStacksAlt    = 0x38;         // [S]
    constexpr auto BuffStacks       = 0x78;         // [S]
}

// ================================================================
// AI MANAGER (Navigation / Pathing)
// ================================================================
namespace AiManager {
    constexpr auto Offset           = 0x41F0;       // [V] LeagueObfuscation offset from IDA sub_28E8C0
    constexpr auto InnerManager     = 0x10;         // [V] Final dereference to real AiManager
    constexpr auto NavPathPtr       = 0x30;         // [S] NavPath pointer (in dec struct)
    constexpr auto TargetPosition   = 0x034;        // [V] Vec3: Click destination / target position
    constexpr auto StartPath        = 0x88;         // [D]
    constexpr auto RefCount         = 0x1F0;        // [S]
    constexpr auto Velocity         = 0x318;        // [V] float: Movement speed value
    constexpr auto IsMoving         = 0x31C;        // [V] bool: Is currently moving
    constexpr auto CurrentSegment   = 0x320;        // [V] int: Current path segment index
    constexpr auto PathStart        = 0x330;        // [V] Vec3: Start of current path
    constexpr auto PathEnd          = 0x33C;        // [V] Vec3: End of current path
    constexpr auto Segments         = 0x348;        // [V] ptr: Waypoints array (Vec3[])
    constexpr auto NavArray         = 0x348;        // [V] ptr: Same as Segments (alias)
    constexpr auto SegmentsCount    = 0x350;        // [V] int: Number of waypoints
    constexpr auto HasPath          = 0x354;        // [V] int: Whether path data exists
    constexpr auto DashSpeed        = 0x360;        // [V] float: Dash speed
    constexpr auto IsDashing        = 0x384;        // [V] bool: Is currently dashing
    constexpr auto TargetPos2       = 0x3A8;        // [V] Vec3: Secondary target position
    constexpr auto ServerPos        = 0x474;        // [V] Vec3: Server-authoritative position
    constexpr auto MoveVec3         = 0x480;        // [S] Vec3: Move direction vector
}

// ================================================================
// HUD INSTANCE
// ================================================================
namespace Hud {
    constexpr auto Camera           = 0x18;         // [S]
    constexpr auto Input            = 0x28;         // [D] oHudMouse
    constexpr auto UserData         = 0x60;         // [S]
    constexpr auto SpellInfo        = 0x68;         // [D] oHudSpell

    // Camera / Zoom
    constexpr auto CameraZoom       = 0x324;        // [IDA] HudCamera + zoom offset
    constexpr auto CameraZoomLimits = 0x310;        // [IDA] ptr to zoom limits struct
    constexpr auto ZoomLimitsMin    = 0x24;         // [IDA] float min zoom in limits struct
    constexpr auto ZoomLimitsMax    = 0x28;         // [IDA] float max zoom in limits struct
    constexpr auto AltZoomLimits    = 0x3D0;        // [IDA] alternate zoom limits
    constexpr auto ZoomLockFlag1    = 0x344;        // [IDA] byte flag zoom lock 1
    constexpr auto ZoomLockFlag2    = 0x345;        // [IDA] byte flag zoom lock 2
 
    // Input / Cursor
    constexpr auto MouseWorldPos    = 0x34;         // [IDA] HudInput + mouse world pos
 
    // User Data
    constexpr auto SelectedObjNetId = 0x28;         // [S]
 
    // Chat  (ChatClient object offsets)
    constexpr auto ChatOpen         = 0x10;         // [IDA] byte flag: 1=chat input active, 0=closed (sub_3B4E00 sets ChatClient+16)
 
    // Viewport W2S
    constexpr auto ViewportW2S      = 0x2B0;        // [IDA] viewport W2S matrix offset
}

// ================================================================
// MISSILE OBJECT
// IDA MCP verified (2026-03-08):
//   sub_886AE0: missile init — copies CastInfo INLINE at missile+0x2C0
//   sub_845A50: CastInfo copy function (full struct layout mapped)
//   sub_90A0E0: missile collision — reads Position at +0x25C, CasterNetId at +0x358
//   sub_49E9F0: returns *(missile+0x128) = SpellData ptr
//   sub_28E710: returns*(missile+0x2C0) = first QWORD = SpellData ptr of CastInfo
//
// CastInfo is INLINE at missile+0x2C0 (NOT a pointer!)
// Read fields directly: startPos = Read<Vec3>(missile + StartPos)
// ================================================================
namespace Missile {
    // --- Missile Object (absolute offsets from missile base) ---
    constexpr auto SpellDataPtr     = 0x128;        // [IDA] sub_49E9F0: *(missile+0x128) = SpellData ptr
    constexpr auto Position         = 0x25C;        // [IDA] sub_90A0E0: Vec3 pos (inherited from GameObject)
    constexpr auto CastInfoBase     = 0x2C0;        // [IDA] sub_886AE0: CastInfo struct INLINE here (NOT a pointer!)
    constexpr auto MissileNetId     = 0x364;        // [IDA] sub_886AE0: [rsi+364h] = NetID (tree key) = CI+0xA4

    // --- CastInfo fields — ABSOLUTE offsets from missile base (0x2C0 + CI_*) ---
    //   Read directly: value = Read<T>(missile + offset)
    constexpr auto CI_SpellData     = 0x2C0;        // [IDA] QWORD: SpellData ptr (CastInfo+0x00)
    constexpr auto SpellName        = 0x2E0;        // [IDA] std::string SSO: spell name (CastInfo+0x20)
    constexpr auto MissileName      = 0x308;        // [IDA] std::string SSO: missile name (CastInfo+0x48)
    constexpr auto StartPos         = 0x388;        // [IDA] Vec3: start position (CastInfo+0xC8)
    constexpr auto EndPos           = 0x394;        // [IDA] Vec3: end position (CastInfo+0xD4)
    constexpr auto CastEndPos       = 0x3A4;        // [IDA] Vec3: cast end position (CastInfo+0xE4)
    constexpr auto CasterNetId      = 0x358;        // [IDA] int: source caster net id (CastInfo+0x98)
    constexpr auto TargetNetId      = 0x35C;        // [IDA] int: target net id (CastInfo+0x9C)
    constexpr auto CI_TargetNetId2  = 0x360;        // [IDA] int: secondary target (CastInfo+0xA0)
    constexpr auto CI_MissileNetId  = 0x364;        // [IDA] int: missile net id (CastInfo+0xA4)
 
    // --- CastInfo relative offsets (for code that needs CI base + offset pattern) ---
    constexpr auto CI_REL_SpellData    = 0x00;      // [IDA] CastInfo+0x00
    constexpr auto CI_REL_SpellName    = 0x20;      // [IDA] CastInfo+0x20
    constexpr auto CI_REL_MissileName  = 0x48;      // [IDA] CastInfo+0x48
    constexpr auto CI_REL_StartPos     = 0xC8;      // [IDA] CastInfo+0xC8
    constexpr auto CI_REL_EndPos       = 0xD4;      // [IDA] CastInfo+0xD4
    constexpr auto CI_REL_CastEndPos   = 0xE4;      // [IDA] CastInfo+0xE4
    constexpr auto CI_REL_CasterNetId  = 0x98;      // [IDA] CastInfo+0x98
    constexpr auto CI_REL_MissileNetId = 0xA4;      // [IDA] CastInfo+0xA4
 
    // --- Legacy aliases ---
    constexpr auto NetworkId        = MissileNetId; // 0x364
    constexpr auto SpellDataInst    = CI_SpellData; // 0x2C0
}

// ================================================================
// BASIC ATTACK / MISC
// ================================================================
namespace BasicAttack {
    constexpr auto Base             = 0x2C68;       // [D]
    constexpr auto Offset1          = 0x2C0;        // [D]
    constexpr auto Offset2          = 0x70;         // [D]
}

namespace Minion {
    constexpr auto LaneArray        = 0x68;         // [D] ptr to lane minion array (relative to MinionManager)
    constexpr auto LaneCount        = 0x70;         // [IDA] count of lane minions (relative to MinionManager)
    constexpr auto LaneType         = 0x4CC9;       // [CE] byte on obj: 4=Melee, 5=Ranged, 6=Cannon, 7=Super
}

// ================================================================
// DRAGON — Offsets for dragon soul detection (IDA sub_456A90 + sub_457DE0)
// ================================================================
namespace Dragon {
    constexpr auto CharacterHash    = 0x68;          // [IDA] DWORD hash on CharacterData (obj+CharData → +0x68)
    // Dragon Name Hash Table (global dword_1D995C0, 9 entries × 40 bytes)
    constexpr auto HashTable        = 0x1D995C0;     // [IDA] static hash table base
    constexpr auto HashTableEnd     = 0x1D99728;     // [IDA] end sentinel
    constexpr auto HashEntrySize    = 0x28;          // 40 bytes per entry (10 DWORDs)
    // Pre-computed dragon name hashes (sub_1074EA0 on dragon names)
    constexpr auto HashAir          = 0x11D34E07;    // SRU_Dragon_Air     → Cloud
    constexpr auto HashFire         = 0x99A9F7D9;    // SRU_Dragon_Fire    → Infernal
    constexpr auto HashWater        = 0x27F69DF4;    // SRU_Dragon_Water   → Ocean
    constexpr auto HashEarth        = 0x606D3187;    // SRU_Dragon_Earth   → Mountain
    constexpr auto HashHextech      = 0xA0808ACE;    // SRU_Dragon_Hextech → Hextech
    constexpr auto HashChemtech     = 0xF94EBA26;    // SRU_Dragon_Chemtech→ Chemtech
    constexpr auto HashRuined       = 0x518A146A;    // SRU_Dragon_Ruined  → Ruined
    constexpr auto HashElder        = 0x5944DC07;    // SRU_Dragon_Elder   → Elder
    constexpr auto HashParty        = 0x4B962AA3;    // SRU_Dragon_Party   → Party
}

// ================================================================
// SPELL CAST INFO (Active Spell)
// From: OnProcessSpell (0x920430) decompilation + chimera
// ================================================================
namespace SpellCastInfo {
    constexpr auto SpellData        = 0x0;          // [IDA] first QWORD = SpellData ptr
    constexpr auto SrcIndex         = 0x98;         // [C] source caster network index
    constexpr auto StartPos         = 0xD8;         // [C] Vec3 spell start position
    constexpr auto EndPos           = 0xE4;         // [C] Vec3 spell end position
    constexpr auto CastPos          = 0xF0;         // [C] Vec3 cast position
    constexpr auto TargetIndex      = 0x108;        // [C] target network index
    constexpr auto CastDelay        = 0x118;        // [C] float cast delay
    constexpr auto IsSpell          = 0x134;        // [C] bool is spell (not auto)
    constexpr auto IsSpecialAttack  = 0x13E;        // [C] bool is special attack
    constexpr auto IsAuto           = 0x141;        // [IDA] byte: is auto attack (chimera=0x13F)
    constexpr auto Slot             = 0x14C;        // [IDA] DWORD: spell slot index (chimera=0x148)
}

// ================================================================
// ITEM SYSTEM
// From: IDA MCP analysis + chimera_structures.h
// ================================================================
namespace ItemSystem {
    // GameObject::ItemList = 0x4D20 (in GameObject namespace)
    // Array of 7 ItemSlot pointers (6 items + trinket)
    constexpr auto SlotInfo         = 0x10;         // [IDA] ItemSlot+0x10 → ItemInfo ptr
    constexpr auto InfoData         = 0x38;         // [IDA] ItemInfo+0x38 → ItemData ptr
    constexpr auto InfoStacks       = 0x64;         // [C] ItemInfo+0x64 → stack count
    constexpr auto DataItemId       = 0xB4;         // [IDA] ItemData+0xB4 → item ID int
    constexpr auto DataAbilityHaste = 0x160;        // [C] ItemData stat
    constexpr auto DataHealth       = 0x164;        // [C] ItemData stat
    constexpr auto DataArmor        = 0x19C;        // [C] ItemData stat
    constexpr auto DataMR           = 0x1BC;        // [C] ItemData stat
    constexpr auto DataAD           = 0x1D8;        // [C] ItemData stat
    constexpr auto DataAP           = 0x1E0;        // [C] ItemData stat
    constexpr auto DataAtkSpeedMult = 0x20C;        // [C] ItemData stat
}

// ================================================================
// NAV GRID
// Source: sig 48 8B 05 ? ? ? ? 0F 28 DA → Global::NavGrid (0x1D7DD08)
// Chain: navGridPtr → +0x8 → NavGridManager → fields below
// IDA MCP verified (2026-03-11): decompile of GetCollisionFlags
// (0x1195B80), sub_1195BC0, sub_1190840, sub_119C040, sub_119C380,
// sub_119C210, sub_119C4F0 — all access *(qword_1D7DD08 + 8) = mgr
//
// KEY FIX: MinX/MinZ were WRONG (0x30/0x38).
// Decompile shows mgr[59] and mgr[61] → float at 59*4=0xEC, 61*4=0xF4
// This was causing intermittent bush/wall detection failure.
// ================================================================
namespace NavGrid {
    // Pointer chain
    constexpr auto NavGridMgr       = 0x8;          // [IDA] navGridPtr → +0x8 → manager

    // Map bounds (float)
    constexpr auto MinX             = 0xEC;         // [IDA] mgr[59] = world min X coordinate
    constexpr auto MinZ             = 0xF4;         // [IDA] mgr[61] = world min Z coordinate
    constexpr auto MaxX             = 0xF8;         // [IDA] mgr[62] = world max X coordinate
    constexpr auto MaxZ             = 0x100;        // [IDA] mgr[64] = world max Z coordinate
 
    // Cell data
    constexpr auto Data             = 0x110;        // [IDA] mgr+272 = ptr to cell array (16 bytes per cell)
    constexpr auto Width            = 0x708;        // [IDA] mgr+1800 = grid width (cells)
    constexpr auto Height           = 0x70C;        // [IDA] mgr+1804 = grid height (cells)
 
    // Scale
    constexpr auto InverseScale     = 0x714;        // [IDA] mgr+1812 = 1/cellSize (MULTIPLY to get cell index)
    constexpr auto Scale            = 0x710;        // [IDA] mgr[452] = cell size (used in bounds check)
 
    // Grass/Brush detection
    constexpr auto GrassRegions     = 0x158;        // [IDA] mgr+344 = grass region bitfield ptr
 
    // Cell structure: 16 bytes per cell
    // Layout: [uint64_t ptrData][uint16_t flags][uint16_t pad][uint32_t pad]
    // If ptrData != 0: real flags = *(uint16_t*)(ptrData + 6)
    // If ptrData == 0: real flags = cell.flags (at cell + 8)
    constexpr auto CellSize         = 16;           // [IDA] bytes per cell
 
    // Collision flag bitmask (from decompile of multiple functions)
    constexpr uint16_t FLAG_WALL    = 0x0001;       // [IDA] sub_119C380: bit 0 = wall
    constexpr uint16_t FLAG_NOWALK  = 0x0002;       // [IDA] sub_119C210: bit 1 = not walkable
    constexpr uint16_t FLAG_BRUSH   = 0x0C00;       // [IDA] sub_119C140: bits 10-11 = brush/grass
    constexpr uint16_t FLAG_SPECIAL = 0x1000;       // [IDA] sub_119C040: bit 12 = special terrain
}

// ================================================================
// MANAGER LIST
// ================================================================
namespace ManagerList {
    constexpr auto Items            = 0x8;          // [S]
    constexpr auto Size             = 0x10;         // [S]
}

// ================================================================
// MINIMAP
// ================================================================
namespace Minimap {
    constexpr auto MinimapParent    = 0x1D7A3D0;    // [CE] global ptr (same as NetInstance)
    constexpr auto MinimapHud       = 0x3B8;         // [CE] MinimapParent->+0x3B8 (was 0x288 in 14.23)
    constexpr auto HudVisible       = 0xD8;          // [CE] MinimapHud+0xD8 byte flag
}

// ================================================================
// EXTRA GLOBALS
// ================================================================
namespace Extra {
    constexpr auto TurretManager    = 0x1D87068;    // [P][IDA] 20 xrefs confirmed
    constexpr auto ViewMatrixInst   = 0x1E2C070;    // [P] view/projection matrix (from offsetplugin.hpp)
    constexpr auto IsClone          = 0x2BB2B0;     // [P] function RVA (+0x10)
}

// ================================================================
// VTABLES
// ================================================================
namespace VTable {
    constexpr auto AIMinionClient   = 0x18DD7F0;    // [P]
}

// ================================================================
// JUNGLE MONSTER NAME STRINGS
// These are string addresses in the binary - version specific!
// Found via IDA MCP find_regex on binary 0x2342000
// NOTE: These are for the IDA binary, NOT the dump binary!
//       For dump binary (0x202D000), re-scan needed.
// ================================================================
namespace JungleNames {
    // IDA binary (0x2342000) string addresses:
    constexpr auto SRU_RiftHerald   = 0x18d5358;    // [IDA] "SRU_RiftHerald"
    constexpr auto SRU_Horde        = 0x18d6690;    // [IDA] "SRU_Horde"
    constexpr auto SRU_Dragon       = 0x18d66B0;    // [IDA] "SRU_Dragon"
    constexpr auto SRU_Dragon_Elder = 0x18d66C0;    // [IDA] "SRU_Dragon_Elder"
    constexpr auto SRU_Baron        = 0x18e58D0;    // [IDA] "SRU_Baron"
}

// ================================================================
// OBJECT TYPE FLAGS (obfuscated field at obj+0x4C)
// Checked via CompareTypeFlags (sub_29CD30) — do NOT read directly!
// Use: Function::CompareTypeFlags(obj, FLAG_xxx)
// Found via IDA MCP decompile of sub_3088A0, sub_308B50, sub_3089A0, sub_308C70
// ================================================================
namespace TypeFlags {
    constexpr auto ObfuscatedField  = 0x4C;          // [IDA] obj+76 in sub_29CD30
    // Bit flags passed to CompareTypeFlags:
    constexpr auto Minion           = 0x0400;         // [IDA] sub_3089A0: flag 1024
    constexpr auto Hero             = 0x0800;         // [IDA] sub_308B50: flag 2048
    constexpr auto JungleMonster    = 0x2000;         // [IDA] sub_3088A0: flag 8192 (IsJungleMonster)
    constexpr auto LargeMonster     = 0x0080;         // [IDA] sub_345650: "Monster_Large" flag
    constexpr auto BuffMonster      = 0x0100;         // [IDA] sub_345650: "Monster_Buff" flag
    constexpr auto MinionSummon     = 0x0100;         // [IDA] sub_345650: "Minion_Summon" flag (same bit)
    constexpr auto Plant            = 0x8000;         // [IDA] sub_345650: "Plant" flag 32768
    constexpr auto CampMonster      = 0x10000;        // [IDA] sub_345650: 0x10000 after Plant
    constexpr auto Crab             = 0x2000;         // [IDA] sub_345650: "Monster_Crab" flag
    constexpr auto IsFleeing        = 0x0200;         // [IDA] sub_345650: fleeing check flag
    constexpr auto AttackableObj    = 0x0008;         // [IDA] sub_345650: attackable
    constexpr auto VisibleObj       = 0x0010;         // [IDA] sub_345650: visible flag
    constexpr auto RenderTarget     = 0x0020;         // [IDA] sub_345650: render target
    constexpr auto IsRecalling      = 0x4000;         // [IDA] sub_345650: recall check
    constexpr auto HasUltimate      = 0x20000;        // [IDA] sub_345650: vtable+2552 check
}

// ================================================================
// MINION CLASSIFICATION (from sub_BBB10 RegisterProperty table)
// LaneMinionType byte value on the minion object, registered via
// sub_10D1B80 with string name + numeric class ID
// Access: use GetJungleType (Function::GetJungleType) or read
//         the byte at the correct offset after finding it at runtime
// Found via IDA MCP decompile of sub_BBB10
// ================================================================
namespace MinionClass {
    // Class IDs (byte values):
    constexpr auto Unset            = 0;              // [IDA] v50=0 "Unset"
    constexpr auto Pet              = 1;              // [IDA] v54=1 "Pet"
    constexpr auto JungleMonster    = 2;              // [IDA] v58=2 "JungleMonster"
    constexpr auto TeamMinion       = 3;              // [IDA] v62=3 "TeamMinion"
    constexpr auto MeleeLaneMinion  = 4;              // [IDA] v66=4 "MeleeLaneMinion"
    constexpr auto RangedLaneMinion = 5;              // [IDA] v70=5 "RangedLaneMinion"
    constexpr auto SiegeLaneMinion  = 6;              // [IDA] v74=6 "SiegeLaneMinion"
    constexpr auto SuperLaneMinion  = 7;              // [IDA] v78=7 "SuperLaneMinion"
}

// ================================================================
// JUNGLE TYPE (from CharacterData sub-object)
// sub_345410 returns *(uint32_t*)(charData + 0x4A84)
// charData = obj + GameObject::CharacterData (0x40C8)
// GetJungleType (sub_66CE60) maps these to:
//   1 → type:0 (Normal),  2 → type:2 (Buff/Dragon), 3 → type:1 (Baron-like)
// Found via IDA MCP decompile of sub_345410 (returns charData+19076)
// ================================================================
namespace JungleType {
    constexpr auto TypeOffset       = 0x4A84;         // [IDA] charData + 19076 in sub_345410

    // Return values from GetJungleType:
    constexpr auto Normal           = 0;              // [IDA] sub_66CE60: case v23-1
    constexpr auto Baron            = 1;              // [IDA] sub_66CE60: v24==0 → return 1
    constexpr auto Dragon           = 2;              // [IDA] sub_66CE60: v22==0 → return 2
}

// ================================================================
// PLANT IDENTIFICATION
// Plants are identified via TypeFlags::Plant (0x8000)
// checked through CompareTypeFlags function
// Plant string names (IDA):
//   "Plant"             @ 0x18EF538
//   "OnPlantActivated"  @ 0x1902660
//   "AttackVisionplant" @ 0x18EBDA0
// Dragon subtypes (IDA string addresses):
//   SRU_Dragon_Air      @ 0x1908F78
//   SRU_Dragon_Fire     @ 0x1908F88
//   SRU_Dragon_Water    @ 0x1908F98
//   SRU_Dragon_Earth    @ 0x1908FB0
//   SRU_Dragon_Ruined   @ 0x1908FC8
//   SRU_Dragon_Hextech  @ 0x1908FE8
//   SRU_Dragon_Chemtech @ 0x1909000
//   SRU_Dragon_Party    @ 0x1909018
// ================================================================
namespace PlantInfo {
    // Plants are checked via: CompareTypeFlags(obj, TypeFlags::Plant)
    // Plant types are distinguished by CharacterName (obj + 0x4330):
    //   "SRU_Plant_Health"   → Honeyfruit (healing plant)
    //   "SRU_Plant_Satchel"  → Blast Cone (knockback plant)
    //   "SRU_Plant_Vision"   → Scryer's Bloom (vision plant)
}

} // namespace Offset
The boss can update a TFT
chen399516 is offline

Old 15th March 2026, 09:15 AM   #12979
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by chen399516 View Post
The boss can update a TFT
<https://www.unknowncheats.me/forum/d...=file&id=53677>
trankhanhtinh1 is offline

Old 15th March 2026, 05:05 PM   #12980
BBasset
Super H4x0r

BBasset's Avatar

Join Date: Aug 2019
Posts: 333
Reputation: 2023
Rep Power: 168
BBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating communityBBasset is a legend in the cheating community
Points: 10,517, Level: 12
Points: 10,517, Level: 12 Points: 10,517, Level: 12 Points: 10,517, Level: 12
Level up: 60%, 483 Points needed
Level up: 60% Level up: 60% Level up: 60%
Activity: 5.3%
Activity: 5.3% Activity: 5.3% Activity: 5.3%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
. . .

bro. doesn't match at all

there is no src's or dest's networkId in missile obj

Code:
        constexpr ULONGLONG SrcIndex = 0x358;              // caster index
        constexpr ULONGLONG DestIndex = 0x3C8;             // [dest ptr] -> target dest index
BBasset is offline

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 649 of 651 « First < 149 549 599 639 645 646 647 648 649 650 651 > 

Tags
typedef, #define, offsets, pobj;, int, float, updated, bool, thread, dword

« Previous Thread | Next Thread »

Forum Jump

    League of Legends

All times are GMT. The time now is 01:45 AM.
Copyright ©2000-2026, Unknowncheats™
DMCA - Contact
Terms of Use - Privacy Policy - Forum Rules

UnknownCheats - Leading the game hacking and cheat development scene since 2000 
UnKnoWnCheaTs Game Hacking Portal UnKnoWnCheaTs Game Hacking Forum – Cheats, Hacks, and Tutorials Download Game Hacks, Cheats and Hacking Tools – UnKnoWnCheaTs Game Hacking Wiki – Tutorials and Guides on UnKnoWnCheaTs Toggle Dark Mode Register at UnKnoWnCheaTs – Join the Greatest Game Hacking Community

AD

Go Back   UnKnoWnCheaTs - Multiplayer Game Hacking and Cheats
MMO and Strategy Games
League of Legends
Reload this Page [Coding] League of Legends Reversal, Structs and Offsets
User Name:
Password:
Remember Me? 

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 650 of 651 « First < 150 550 600 640 646 647 648 649 650 651 > 

Thread Tools
Old 15th March 2026, 06:30 PM   #12981
rabbit315
Super l337

rabbit315's Avatar

Join Date: Jan 2023
Posts: 217
Reputation: 10
Rep Power: 81
rabbit315 has made posts that are generally average in quality
Points: 3,917, Level: 6
Points: 3,917, Level: 6 Points: 3,917, Level: 6 Points: 3,917, Level: 6
Level up: 36%, 583 Points needed
Level up: 36% Level up: 36% Level up: 36%
Activity: 15.8%
Activity: 15.8% Activity: 15.8% Activity: 15.8%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
How can I detect if a casted spell has a progress bar (e.g., Varus Q, Ezreal R)?

My Orbwalker is pretty much perfect, but when I use Varus Q, the Orbwalker is still running. I want to find a way to filter these spells out.
rabbit315 is offline

AD

Old 16th March 2026, 04:41 AM   #12982
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by BBasset View Post
bro. doesn't match at all

there is no src's or dest's networkId in missile obj

Code:
        constexpr ULONGLONG SrcIndex = 0x358;              // caster index
        constexpr ULONGLONG DestIndex = 0x3C8;             // [dest ptr] -> target dest index
Oh, that was my oversight, I'll update it, thanks for the feedback.
trankhanhtinh1 is offline

Old 16th March 2026, 06:36 AM   #12983
msfool
Senior Member

msfool's Avatar

Join Date: Jun 2023
Posts: 81
Reputation: 10
Rep Power: 68
msfool has made posts that are generally average in quality
Points: 2,236, Level: 4
Points: 2,236, Level: 4 Points: 2,236, Level: 4 Points: 2,236, Level: 4
Level up: 20%, 564 Points needed
Level up: 20% Level up: 20% Level up: 20%
Activity: 5.0%
Activity: 5.0% Activity: 5.0% Activity: 5.0%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by Alexis913 View Post
That doesn't work because it gives you 2 results.

I use this pattern:
48 8B 3D ?? ?? ?? ?? FF CA
maybe you are right,it's CN server sig
msfool is offline

Old 16th March 2026, 11:01 AM   #12984
hernos
Banned

hernos's Avatar

Join Date: Nov 2013
Location: #BringBackGithub
Posts: 2,055
Reputation: 24708
Rep Power: 0
hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!hernos has reputation that takes up 2GB of server space!
Points: 47,416, Level: 32
Points: 47,416, Level: 32 Points: 47,416, Level: 32 Points: 47,416, Level: 32
Level up: 83%, 584 Points needed
Level up: 83% Level up: 83% Level up: 83%
Activity: 47.5%
Activity: 47.5% Activity: 47.5% Activity: 47.5%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by rabbit315 View Post
How can I detect if a casted spell has a progress bar (e.g., Varus Q, Ezreal R)?

My Orbwalker is pretty much perfect, but when I use Varus Q, the Orbwalker is still running. I want to find a way to filter these spells out.
CastDelay > 0.0f
hernos is offline

Old 16th March 2026, 08:11 PM   #12985
rabbit315
Super l337

rabbit315's Avatar

Join Date: Jan 2023
Posts: 217
Reputation: 10
Rep Power: 81
rabbit315 has made posts that are generally average in quality
Points: 3,917, Level: 6
Points: 3,917, Level: 6 Points: 3,917, Level: 6 Points: 3,917, Level: 6
Level up: 36%, 583 Points needed
Level up: 36% Level up: 36% Level up: 36%
Activity: 15.8%
Activity: 15.8% Activity: 15.8% Activity: 15.8%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);

Has mouse_event lost its effectiveness? After I updated my project, I found that the movement is no longer working

Anyone got a sig for TryRightClick？
Last edited by rabbit315; 18th March 2026 at 05:01 AM.
rabbit315 is offline

Old 18th March 2026, 10:40 AM   #12986
woaicao13
n00bie

woaicao13's Avatar

Join Date: Mar 2026
Posts: 2
Reputation: 10
Rep Power: 2
woaicao13 has made posts that are generally average in quality
Points: 14, Level: 1
Points: 14, Level: 1 Points: 14, Level: 1 Points: 14, Level: 1
Level up: 4%, 386 Points needed
Level up: 4% Level up: 4% Level up: 4%
Activity: 2.6%
Activity: 2.6% Activity: 2.6% Activity: 2.6%
Guys, could you please tell me what the latest R3nz skin offset base address is? Thank you very much. I will wait for your message.Thank you again.
woaicao13 is offline

Old 19th March 2026, 03:25 AM   #12987
woaicao13
n00bie

woaicao13's Avatar

Join Date: Mar 2026
Posts: 2
Reputation: 10
Rep Power: 2
woaicao13 has made posts that are generally average in quality
Points: 14, Level: 1
Points: 14, Level: 1 Points: 14, Level: 1 Points: 14, Level: 1
Level up: 4%, 386 Points needed
Level up: 4% Level up: 4% Level up: 4%
Activity: 2.6%
Activity: 2.6% Activity: 2.6% Activity: 2.6%
Guys, I have obtained the data from Hydy100. Thank you.
woaicao13 is offline

Old 21st March 2026, 05:06 AM   #12988
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
new offset for hotfix

Code:
# pragma once

// Generated by tools/dump_object_family_offsets_ida.py
// Generated UTC: 2026-03-21T05:04:47+00:00
// Seed header: D:\source\LOL_Dumper_[unknowncheats.me]_\EnsoulSharp.SDK-master\ImGui-DirectX-11-Kiero-Hook-master\Nightsharp\core\Offsets.h
// Input file: C:\Users\MR THINH\Downloads\dump\League of Legends_exe_PID3a84_League of Legends.exe_7FF6D05B0000_x64.exe
// Module: League of Legends_exe_PID3a84_League of Legends.exe_7FF6D05B0000_x64.exe

namespace Offset {

// ----------------------------------------------------------------
// EnsoulSharp-style runtime API groups
// ----------------------------------------------------------------

namespace GameObjectsRuntime {
    constexpr auto Player = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto Objects = 0x1DA1488; // [SEED]; Offsets.h::Global::ObjectManager; via Runtime.Global::ObjectManager
    constexpr auto Heroes = 0x1DA14E0; // [SEED]; Offsets.h::Global::HeroManager; via Runtime.Global::HeroManager
    constexpr auto Minions = 0x1DA14D8; // [SEED]; Offsets.h::Global::MinionManager; via Runtime.Global::MinionManager
    constexpr auto Missiles = 0x1DA5270; // [SEED]; Offsets.h::Global::MissileManager; via Runtime.Global::MissileManager
    constexpr auto Turrets = 0x1DAE248; // [SEED]; Offsets.h::Global::TurretManager; via Runtime.Global::TurretManager
    constexpr auto UnderMouseObject = 0x19ECD78; // [SEED]; Offsets.h::Global::UnderMouseObj; via Runtime.Global::UnderMouseObj
} // namespace GameObjectsRuntime

namespace ObjectManagerRuntime {
    constexpr auto Player = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto ObjectManager = 0x1DA1488; // [SEED]; Offsets.h::Global::ObjectManager; via Runtime.Global::ObjectManager
    constexpr auto HeroManager = 0x1DA14E0; // [SEED]; Offsets.h::Global::HeroManager; via Runtime.Global::HeroManager
    constexpr auto MinionManager = 0x1DA14D8; // [SEED]; Offsets.h::Global::MinionManager; via Runtime.Global::MinionManager
    constexpr auto MissileManager = 0x1DA5270; // [SEED]; Offsets.h::Global::MissileManager; via Runtime.Global::MissileManager
    constexpr auto TurretManager = 0x1DAE248; // [SEED]; Offsets.h::Global::TurretManager; via Runtime.Global::TurretManager
    constexpr auto ManagerListItems = 0x8; // [S]; stable layout; via Runtime.ManagerList::Items
    constexpr auto ManagerListSize = 0x10; // [S]; stable layout; via Runtime.ManagerList::Size
    constexpr auto GetFirstObject = 0x9C39B0; // [SEED]; Offsets.h::Function::GetFirstObject; via Runtime.Function::GetFirstObject
    constexpr auto GetFirstObjectAlt = 0x9C39B0; // [SEED]; Offsets.h::Function::GetFirstObjectAlt; via Runtime.Function::GetFirstObjectAlt
    constexpr auto GetNextObject = 0x523760; // [SEED]; Offsets.h::Function::GetNextObject; via Runtime.Function::GetNextObject
    constexpr auto FindObject = 0x522530; // [SEED]; Offsets.h::Function::FindObject; via Runtime.Function::FindObject
} // namespace ObjectManagerRuntime

namespace GameRuntime {
    constexpr auto LocalPlayer = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto GameTime = 0x1DAF720; // [SEED]; Offsets.h::Global::GameTime; via Runtime.Global::GameTime
    constexpr auto NetInstance = 0x1DA1480; // [SEED]; Offsets.h::Global::NetInstance; via Runtime.Global::NetInstance
    constexpr auto ChatClient = 0x1DB43E0; // [SEED]; Offsets.h::Global::ChatClient; via Runtime.Global::ChatClient
    constexpr auto ChatInstance = 0x1DA5480; // [SEED]; Offsets.h::Global::ChatInstance; via Runtime.Global::ChatInstance
    constexpr auto ShopInstance = 0x1DB43F8; // [SEED]; Offsets.h::Global::ShopInstance; via Runtime.Global::ShopInstance
    constexpr auto OpenWindowsArray = 0x1E66E78; // [SEED]; Offsets.h::Global::OpenWindowsArray; via Runtime.Global::OpenWindowsArray
    constexpr auto OpenWindowsCount = 0x1E66E80; // [SEED]; Offsets.h::Global::OpenWindowsCount; via Runtime.Global::OpenWindowsCount
    constexpr auto CursorPosRaw = 0x1E2DC38; // [SEED]; Offsets.h::Global::CursorInstance; via Runtime.Global::CursorInstance
    constexpr auto MouseScreenVec2 = 0x1DA5218; // [SEED]; Offsets.h::Global::MouseScreenVec2; via Runtime.Global::MouseScreenVec2
    constexpr auto UnderMouseObject = 0x19ECD78; // [SEED]; Offsets.h::Global::UnderMouseObj; via Runtime.Global::UnderMouseObj
    constexpr auto GetPing = 0x677420; // [SEED]; Offsets.h::Function::GetPing; via Runtime.Function::GetPing
    constexpr auto GetMapID = 0x2933B0; // [SEED]; Offsets.h::Function::GetMapID; via Runtime.Function::GetMapID
    constexpr auto PrintChat = 0x10B11B0; // [SEED]; Offsets.h::Function::PrintChat; via Runtime.Function::PrintChat
} // namespace GameRuntime

namespace DrawingRuntime {
    constexpr auto WorldToScreen = 0x1260DC0; // [SEED]; Offsets.h::Function::WorldToScreen; via Runtime.Function::WorldToScreen
    constexpr auto HudInstance = 0x1DA1628; // [SEED]; Offsets.h::Global::HudInstance; via Runtime.Global::HudInstance
    constexpr auto ViewPort = 0x1DB4398; // [SEED]; Offsets.h::Global::ViewPort; via Runtime.Global::ViewPort
    constexpr auto ViewPort2 = 0x1E68458; // [SEED]; Offsets.h::Global::ViewPort2; via Runtime.Global::ViewPort2
    constexpr auto Renderer = 0x1E68450; // [SEED]; Offsets.h::Global::r3dRenderer; via Runtime.Global::r3dRenderer
    constexpr auto Camera = 0x18; // [SEED]; Offsets.h::Hud::Camera; via Runtime.Hud::Camera
    constexpr auto Input = 0x28; // [SEED]; Offsets.h::Hud::Input; via Runtime.Hud::Input
    constexpr auto UserData = 0x60; // [SEED]; Offsets.h::Hud::UserData; via Runtime.Hud::UserData
    constexpr auto SpellInfo = 0x68; // [SEED]; Offsets.h::Hud::SpellInfo; via Runtime.Hud::SpellInfo
    constexpr auto CameraZoom = 0x324; // [SEED]; Offsets.h::Hud::CameraZoom; via Runtime.Hud::CameraZoom
    constexpr auto CameraZoomLimits = 0x310; // [SEED]; Offsets.h::Hud::CameraZoomLimits; via Runtime.Hud::CameraZoomLimits
    constexpr auto ZoomLimitsMin = 0x24; // [SEED]; Offsets.h::Hud::ZoomLimitsMin; via Runtime.Hud::ZoomLimitsMin
    constexpr auto ZoomLimitsMax = 0x28; // [SEED]; Offsets.h::Hud::ZoomLimitsMax; via Runtime.Hud::ZoomLimitsMax
    constexpr auto AltZoomLimits = 0x3D0; // [SEED]; Offsets.h::Hud::AltZoomLimits; via Runtime.Hud::AltZoomLimits
    constexpr auto ZoomLockFlag1 = 0x344; // [SEED]; Offsets.h::Hud::ZoomLockFlag1; via Runtime.Hud::ZoomLockFlag1
    constexpr auto ZoomLockFlag2 = 0x345; // [SEED]; Offsets.h::Hud::ZoomLockFlag2; via Runtime.Hud::ZoomLockFlag2
    constexpr auto MouseWorldPos = 0x34; // [SEED]; Offsets.h::Hud::MouseWorldPos; via Runtime.Hud::MouseWorldPos
    constexpr auto SelectedObjNetId = 0x28; // [SEED]; Offsets.h::Hud::SelectedObjNetId; via Runtime.Hud::SelectedObjNetId
    constexpr auto ChatOpen = 0x10; // [SEED]; Offsets.h::Hud::ChatOpen; via Runtime.Hud::ChatOpen
    constexpr auto ViewportW2S = 0x2B0; // [SEED]; Offsets.h::Hud::ViewportW2S; via Runtime.Hud::ViewportW2S
} // namespace DrawingRuntime

namespace ControlRuntime {
    constexpr auto IssueOrder = 0x2A5040; // [SEED]; Offsets.h::Function::IssueOrder -> IssueOrderCore; via Runtime.Function::IssueOrder
    constexpr auto IssueOrderCore = 0x2A5040; // [SEED]; Offsets.h::Function::IssueOrderCore; via Runtime.Function::IssueOrderCore
    constexpr auto CastSpellSafe = 0xBB8950; // [SEED]; Offsets.h::Function::CastSpellSafe; via Runtime.Function::CastSpellSafe
    constexpr auto GetSpellCastInfo = 0x288D50; // [SEED]; Offsets.h::Function::GetSpellCastInfo; via Runtime.Function::GetSpellCastInfo
    constexpr auto GetSpellSlot = 0x905BC0; // [SEED]; Offsets.h::Function::GetSpellSlot; via Runtime.Function::GetSpellSlot
    constexpr auto GetResourceType = 0x286070; // [SEED]; Offsets.h::Function::GetResourceType; via Runtime.Function::GetResourceType
    constexpr auto GetAttackDelay = 0x53A3C0; // [SEED]; Offsets.h::Function::GetAttackDelay; via Runtime.Function::GetAttackDelay
    constexpr auto GetAttackWindup = 0x53A2C0; // [SEED]; Offsets.h::Function::GetAttackWindup; via Runtime.Function::GetAttackWindup
    constexpr auto GetBoundingRadius = 0x28A600; // [SEED]; Offsets.h::Function::GetBoundingRadius; via Runtime.Function::GetBoundingRadius
    constexpr auto IssueOrderFlag = 0x1D04FA8; // [SEED]; Offsets.h::Flag::IssueOrderFlag; via Runtime.Flag::IssueOrderFlag
    constexpr auto CastSpellFlag = 0x1D04F40; // [SEED]; Offsets.h::Flag::CastSpellFlag; via Runtime.Flag::CastSpellFlag
} // namespace ControlRuntime

namespace EventRuntime {
    constexpr auto CreateClientEffect = 0x83C170; // [SEED]; Offsets.h::Function::CreateClientEffect; via Runtime.Function::CreateClientEffect
    constexpr auto OnCreateObject = 0x527930; // [SEED]; Offsets.h::Function::OnCreateObject; via Runtime.Function::OnCreateObject
    constexpr auto OnGameUpdate = 0x5215E0; // [SEED]; Offsets.h::Function::OnGameUpdate; via Runtime.Function::OnGameUpdate
    constexpr auto OnProcessSpell = 0x91D1B0; // [SEED]; Offsets.h::Function::OnProcessSpell; via Runtime.Function::OnProcessSpell
    constexpr auto OnSpellImpact = 0x914320; // [SEED]; Offsets.h::Function::OnSpellImpact; via Runtime.Function::OnSpellImpact
    constexpr auto OnStopCast = 0x91D750; // [SEED]; Offsets.h::Function::OnStopCast; via Runtime.Function::OnStopCast
    constexpr auto OnFinishCast = 0x2CBE30; // [SEED]; Offsets.h::Function::OnFinishCast; via Runtime.Function::OnFinishCast
    constexpr auto OnBuffAdd = 0xBD0B40; // [SEED]; Offsets.h::Function::OnBuffAdd; via Runtime.Function::OnBuffAdd
} // namespace EventRuntime

namespace NavGridRuntime {
    constexpr auto NavGrid = 0x1DA51E0; // [SEED]; Offsets.h::Global::NavGrid; via Runtime.Global::NavGrid
    constexpr auto GetCollisionFlags = 0x11B29D0; // [SEED]; Offsets.h::Function::GetCollisionFlags; via Runtime.Function::GetCollisionFlags
    constexpr auto GetAiManager = 0x292420; // [SEED]; Offsets.h::Function::GetAiManager; via Runtime.Function::GetAiManager
    constexpr auto GetAiManagerInner = 0x293A10; // [SEED]; Offsets.h::Function::GetAiManagerInner; via Runtime.Function::GetAiManagerInner
    constexpr auto NavGridMgr = 0x8; // [SEED]; Offsets.h::NavGrid::NavGridMgr; via Runtime.NavGrid::NavGridMgr
    constexpr auto MinX = 0xEC; // [SEED]; Offsets.h::NavGrid::MinX; via Runtime.NavGrid::MinX
    constexpr auto MinZ = 0xF4; // [SEED]; Offsets.h::NavGrid::MinZ; via Runtime.NavGrid::MinZ
    constexpr auto MaxX = 0xF8; // [SEED]; Offsets.h::NavGrid::MaxX; via Runtime.NavGrid::MaxX
    constexpr auto MaxZ = 0x100; // [SEED]; Offsets.h::NavGrid::MaxZ; via Runtime.NavGrid::MaxZ
    constexpr auto Data = 0x110; // [SEED]; Offsets.h::NavGrid::Data; via Runtime.NavGrid::Data
    constexpr auto Width = 0x708; // [SEED]; Offsets.h::NavGrid::Width; via Runtime.NavGrid::Width
    constexpr auto Height = 0x70C; // [SEED]; Offsets.h::NavGrid::Height; via Runtime.NavGrid::Height
    constexpr auto Scale = 0x710; // [SEED]; Offsets.h::NavGrid::Scale; via Runtime.NavGrid::Scale
    constexpr auto InverseScale = 0x714; // [SEED]; Offsets.h::NavGrid::InverseScale; via Runtime.NavGrid::InverseScale
    constexpr auto GrassRegions = 0x158; // [SEED]; Offsets.h::NavGrid::GrassRegions; via Runtime.NavGrid::GrassRegions
    constexpr auto CellSize = 0x10; // [SEED]; Offsets.h::NavGrid::CellSize; via Runtime.NavGrid::CellSize
    constexpr auto FlagWall = 0x1; // [SEED]; Offsets.h::NavGrid::FLAG_WALL; via Runtime.NavGrid::FLAG_WALL
    constexpr auto FlagNoWalk = 0x2; // [SEED]; Offsets.h::NavGrid::FLAG_NOWALK; via Runtime.NavGrid::FLAG_NOWALK
    constexpr auto FlagBrush = 0xC00; // [SEED]; Offsets.h::NavGrid::FLAG_BRUSH; via Runtime.NavGrid::FLAG_BRUSH
    constexpr auto FlagSpecial = 0x1000; // [SEED]; Offsets.h::NavGrid::FLAG_SPECIAL; via Runtime.NavGrid::FLAG_SPECIAL
} // namespace NavGridRuntime

namespace SpellRuntime {
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto SpellCastInfoSpellData = 0x0; // [SEED]; Offsets.h::SpellCastInfo::SpellData; via Runtime.SpellCastInfo::SpellData
    constexpr auto SpellCastInfoSrcIndex = 0x98; // [S]; stable layout; via Runtime.SpellCastInfo::SrcIndex
    constexpr auto SpellCastInfoStartPos = 0xD8; // [S]; stable layout; via Runtime.SpellCastInfo::StartPos
    constexpr auto SpellCastInfoEndPos = 0xE4; // [S]; stable layout; via Runtime.SpellCastInfo::EndPos
    constexpr auto SpellCastInfoCastPos = 0xF0; // [S]; stable layout; via Runtime.SpellCastInfo::CastPos
    constexpr auto SpellCastInfoTargetIndex = 0x108; // [S]; stable layout; via Runtime.SpellCastInfo::TargetIndex
    constexpr auto SpellCastInfoDestIndex = 0x108; // [ALIAS]; alias of TargetIndex; via Runtime.SpellCastInfo::DestIndex
    constexpr auto SpellCastInfoCastDelay = 0x118; // [S]; stable layout; via Runtime.SpellCastInfo::CastDelay
    constexpr auto SpellCastInfoIsSpell = 0x134; // [SEED]; Offsets.h::SpellCastInfo::IsSpell; via Runtime.SpellCastInfo::IsSpell
    constexpr auto SpellCastInfoIsSpecialAttack = 0x13E; // [SEED]; Offsets.h::SpellCastInfo::IsSpecialAttack; via Runtime.SpellCastInfo::IsSpecialAttack
    constexpr auto SpellCastInfoIsAuto = 0x141; // [S]; stable layout; via Runtime.SpellCastInfo::IsAuto
    constexpr auto SpellCastInfoSlot = 0x14C; // [S]; stable layout; via Runtime.SpellCastInfo::Slot
} // namespace SpellRuntime

namespace ItemRuntime {
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
    constexpr auto SlotInfo = 0x10; // [S]; stable layout; via Runtime.ItemSystem::SlotInfo
    constexpr auto InfoData = 0x38; // [S]; stable layout; via Runtime.ItemSystem::InfoData
    constexpr auto InfoStacks = 0x64; // [S]; stable layout; via Runtime.ItemSystem::InfoStacks
    constexpr auto DataItemId = 0xB4; // [SEED]; Offsets.h::ItemSystem::DataItemId; via Runtime.ItemSystem::DataItemId
    constexpr auto DataAbilityHaste = 0x160; // [SEED]; Offsets.h::ItemSystem::DataAbilityHaste; via Runtime.ItemSystem::DataAbilityHaste
    constexpr auto DataHealth = 0x164; // [SEED]; Offsets.h::ItemSystem::DataHealth; via Runtime.ItemSystem::DataHealth
    constexpr auto DataArmor = 0x19C; // [SEED]; Offsets.h::ItemSystem::DataArmor; via Runtime.ItemSystem::DataArmor
    constexpr auto DataMR = 0x1BC; // [SEED]; Offsets.h::ItemSystem::DataMR; via Runtime.ItemSystem::DataMR
    constexpr auto DataAD = 0x1D8; // [SEED]; Offsets.h::ItemSystem::DataAD; via Runtime.ItemSystem::DataAD
    constexpr auto DataAP = 0x1E0; // [SEED]; Offsets.h::ItemSystem::DataAP; via Runtime.ItemSystem::DataAP
    constexpr auto DataAtkSpeedMult = 0x20C; // [SEED]; Offsets.h::ItemSystem::DataAtkSpeedMult; via Runtime.ItemSystem::DataAtkSpeedMult
} // namespace ItemRuntime

namespace ClassificationRuntime {
    constexpr auto TypeFlagsField = 0x4C; // [SEED]; Offsets.h::TypeFlags::ObfuscatedField; via Runtime.TypeFlags::ObfuscatedField
    constexpr auto TypeIsObjectAI = 0x400; // [SEED]; Offsets.h::TypeFlags::IsObjectAI; via Runtime.TypeFlags::IsObjectAI
    constexpr auto TypeMinion = 0x800; // [SEED]; Offsets.h::TypeFlags::Minion; via Runtime.TypeFlags::Minion
    constexpr auto TypeHero = 0x1000; // [SEED]; Offsets.h::TypeFlags::Hero; via Runtime.TypeFlags::Hero
    constexpr auto TypeTurret = 0x2000; // [SEED]; Offsets.h::TypeFlags::Turret; via Runtime.TypeFlags::Turret
    constexpr auto TypePlant = 0x8000; // [SEED]; Offsets.h::TypeFlags::Plant; via Runtime.TypeFlags::Plant
    constexpr auto TypeLargeMonster = 0x80; // [SEED]; Offsets.h::TypeFlags::LargeMonster; via Runtime.TypeFlags::LargeMonster
    constexpr auto TypeBuffMonster = 0x100; // [SEED]; Offsets.h::TypeFlags::BuffMonster; via Runtime.TypeFlags::BuffMonster
    constexpr auto TypeMinionSummon = 0x100; // [SEED]; Offsets.h::TypeFlags::MinionSummon; via Runtime.TypeFlags::MinionSummon
    constexpr auto TypeAttackableObj = 0x8; // [SEED]; Offsets.h::TypeFlags::AttackableObj; via Runtime.TypeFlags::AttackableObj
    constexpr auto TypeVisibleObj = 0x10; // [SEED]; Offsets.h::TypeFlags::VisibleObj; via Runtime.TypeFlags::VisibleObj
    constexpr auto TypeRenderTarget = 0x20; // [SEED]; Offsets.h::TypeFlags::RenderTarget; via Runtime.TypeFlags::RenderTarget
    constexpr auto TypeIsRecalling = 0x4000; // [SEED]; Offsets.h::TypeFlags::IsRecalling; via Runtime.TypeFlags::IsRecalling
    constexpr auto MinionLaneArray = 0x68; // [SEED]; Offsets.h::Minion::LaneArray; via Runtime.Minion::LaneArray
    constexpr auto MinionLaneCount = 0x70; // [SEED]; Offsets.h::Minion::LaneCount; via Runtime.Minion::LaneCount
    constexpr auto MinionLaneType = 0x4CC9; // [SEED]; Offsets.h::Minion::LaneType; via Runtime.Minion::LaneType
    constexpr auto MinionClassUnset = 0x0; // [SEED]; Offsets.h::MinionClass::Unset; via Runtime.MinionClass::Unset
    constexpr auto MinionClassPet = 0x1; // [SEED]; Offsets.h::MinionClass::Pet; via Runtime.MinionClass::Pet
    constexpr auto MinionClassJungleMonster = 0x2; // [SEED]; Offsets.h::MinionClass::JungleMonster; via Runtime.MinionClass::JungleMonster
    constexpr auto MinionClassTeamMinion = 0x3; // [SEED]; Offsets.h::MinionClass::TeamMinion; via Runtime.MinionClass::TeamMinion
    constexpr auto MinionClassMeleeLaneMinion = 0x4; // [SEED]; Offsets.h::MinionClass::MeleeLaneMinion; via Runtime.MinionClass::MeleeLaneMinion
    constexpr auto MinionClassRangedLaneMinion = 0x5; // [SEED]; Offsets.h::MinionClass::RangedLaneMinion; via Runtime.MinionClass::RangedLaneMinion
    constexpr auto MinionClassSiegeLaneMinion = 0x6; // [SEED]; Offsets.h::MinionClass::SiegeLaneMinion; via Runtime.MinionClass::SiegeLaneMinion
    constexpr auto MinionClassSuperLaneMinion = 0x7; // [SEED]; Offsets.h::MinionClass::SuperLaneMinion; via Runtime.MinionClass::SuperLaneMinion
    constexpr auto JungleTypeOffset = 0x4A84; // [SEED]; Offsets.h::JungleType::TypeOffset; via Runtime.JungleType::TypeOffset
    constexpr auto JungleTypeNormal = 0x0; // [SEED]; Offsets.h::JungleType::Normal; via Runtime.JungleType::Normal
    constexpr auto JungleTypeBaron = 0x1; // [SEED]; Offsets.h::JungleType::Baron; via Runtime.JungleType::Baron
    constexpr auto JungleTypeDragon = 0x2; // [SEED]; Offsets.h::JungleType::Dragon; via Runtime.JungleType::Dragon
} // namespace ClassificationRuntime

// ----------------------------------------------------------------
// EnsoulSharp-style object family groups
// ----------------------------------------------------------------

namespace All {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
} // namespace All

namespace AttackableUnit {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
} // namespace AttackableUnit

namespace AIHeroClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto MP = 0x360; // [SEED]; Offsets.h::Mana::MP; via AIBaseClientMana::MP
    constexpr auto MaxMP = 0x388; // [SEED]; Offsets.h::Mana::MaxMP; via AIBaseClientMana::MaxMP
    constexpr auto PAR = 0xE00; // [SEED]; Offsets.h::Mana::PAR; via AIBaseClientMana::PAR
    constexpr auto MaxPAR = 0xE28; // [INFERRED]; PAR + 0x28 inferred LeagueObfuscation pair for MaxPAR; via AIBaseClientMana::MaxPAR
    constexpr auto SAR = 0x108; // [SEED]; Offsets.h::Mana::SAR; via AIBaseClientMana::SAR
    constexpr auto MaxSAR = 0x130; // [SEED]; Offsets.h::Mana::MaxSAR; via AIBaseClientMana::MaxSAR
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto HeroStatsBase = 0x1B88; // [SEED]; Offsets.h::HeroStats::Base; via AIBaseClientHeroStats::Base
    constexpr auto PercentCooldownMod = 0x1B88; // [SEED]; Offsets.h::HeroStats::PercentCooldownMod; via AIBaseClientHeroStats::PercentCooldownMod
    constexpr auto AbilityHaste = 0x1BB0; // [SEED]; Offsets.h::HeroStats::AbilityHaste; via AIBaseClientHeroStats::AbilityHaste
    constexpr auto PercentCooldownCapMod = 0x1BD8; // [SEED]; Offsets.h::HeroStats::PercentCooldownCapMod; via AIBaseClientHeroStats::PercentCooldownCapMod
    constexpr auto PassiveCdEndTime = 0x1C00; // [SEED]; Offsets.h::HeroStats::PassiveCdEndTime; via AIBaseClientHeroStats::PassiveCdEndTime
    constexpr auto PassiveCdTotalTime = 0x1C28; // [SEED]; Offsets.h::HeroStats::PassiveCdTotalTime; via AIBaseClientHeroStats::PassiveCdTotalTime
    constexpr auto FlatPhysicalDmgMod = 0x1CC8; // [SEED]; Offsets.h::HeroStats::FlatPhysicalDmgMod; via AIBaseClientHeroStats::FlatPhysicalDmgMod
    constexpr auto PercentPhysicalDmgMod = 0x1CF0; // [SEED]; Offsets.h::HeroStats::PercentPhysicalDmgMod; via AIBaseClientHeroStats::PercentPhysicalDmgMod
    constexpr auto PercentBonusPhysDmgMod = 0x1D18; // [SEED]; Offsets.h::HeroStats::PercentBonusPhysDmgMod; via AIBaseClientHeroStats::PercentBonusPhysDmgMod
    constexpr auto PercentBasePhysDmgFlat = 0x1D40; // [SEED]; Offsets.h::HeroStats::PercentBasePhysDmgFlat; via AIBaseClientHeroStats::PercentBasePhysDmgFlat
    constexpr auto FlatMagicDmgMod = 0x1D68; // [SEED]; Offsets.h::HeroStats::FlatMagicDmgMod; via AIBaseClientHeroStats::FlatMagicDmgMod
    constexpr auto PercentMagicDmgMod = 0x1D90; // [SEED]; Offsets.h::HeroStats::PercentMagicDmgMod; via AIBaseClientHeroStats::PercentMagicDmgMod
    constexpr auto FlatMagicReduction = 0x1DB8; // [SEED]; Offsets.h::HeroStats::FlatMagicReduction; via AIBaseClientHeroStats::FlatMagicReduction
    constexpr auto PercentMagicReduction = 0x1DE0; // [SEED]; Offsets.h::HeroStats::PercentMagicReduction; via AIBaseClientHeroStats::PercentMagicReduction
    constexpr auto FlatCastRangeMod = 0x1E08; // [SEED]; Offsets.h::HeroStats::FlatCastRangeMod; via AIBaseClientHeroStats::FlatCastRangeMod
    constexpr auto AttackSpeedMod = 0x1E30; // [SEED]; Offsets.h::HeroStats::AttackSpeedMod; via AIBaseClientHeroStats::AttackSpeedMod
    constexpr auto PercentAttackSpeedMod = 0x1E58; // [SEED]; Offsets.h::HeroStats::PercentAttackSpeedMod; via AIBaseClientHeroStats::PercentAttackSpeedMod
    constexpr auto PercentMultiAtkSpeedMod = 0x1E80; // [SEED]; Offsets.h::HeroStats::PercentMultiAtkSpeedMod; via AIBaseClientHeroStats::PercentMultiAtkSpeedMod
    constexpr auto PercentHealingAmountMod = 0x1EA8; // [SEED]; Offsets.h::HeroStats::PercentHealingAmountMod; via AIBaseClientHeroStats::PercentHealingAmountMod
    constexpr auto BaseAttackDamage = 0x1ED0; // [SEED]; Offsets.h::HeroStats::BaseAttackDamage; via AIBaseClientHeroStats::BaseAttackDamage
    constexpr auto BaseAtkDmgSansScale = 0x1EF8; // [SEED]; Offsets.h::HeroStats::BaseAtkDmgSansScale; via AIBaseClientHeroStats::BaseAtkDmgSansScale
    constexpr auto FlatBaseAtkDmgMod = 0x1F20; // [SEED]; Offsets.h::HeroStats::FlatBaseAtkDmgMod; via AIBaseClientHeroStats::FlatBaseAtkDmgMod
    constexpr auto PercentBaseAtkDmgMod = 0x1F48; // [SEED]; Offsets.h::HeroStats::PercentBaseAtkDmgMod; via AIBaseClientHeroStats::PercentBaseAtkDmgMod
    constexpr auto BaseAbilityDamage = 0x1F70; // [SEED]; Offsets.h::HeroStats::BaseAbilityDamage; via AIBaseClientHeroStats::BaseAbilityDamage
    constexpr auto CritDamageMultiplier = 0x1F98; // [SEED]; Offsets.h::HeroStats::CritDamageMultiplier; via AIBaseClientHeroStats::CritDamageMultiplier
    constexpr auto ScaleSkinCoef = 0x1FC0; // [SEED]; Offsets.h::HeroStats::ScaleSkinCoef; via AIBaseClientHeroStats::ScaleSkinCoef
    constexpr auto Dodge = 0x1FE8; // [SEED]; Offsets.h::HeroStats::Dodge; via AIBaseClientHeroStats::Dodge
    constexpr auto Crit = 0x2010; // [SEED]; Offsets.h::HeroStats::Crit; via AIBaseClientHeroStats::Crit
    constexpr auto Armor = 0x2060; // [SEED]; Offsets.h::HeroStats::Armor; via AIBaseClientHeroStats::Armor
    constexpr auto BonusArmor = 0x2088; // [SEED]; Offsets.h::HeroStats::BonusArmor; via AIBaseClientHeroStats::BonusArmor
    constexpr auto SpellBlock = 0x20B0; // [SEED]; Offsets.h::HeroStats::SpellBlock; via AIBaseClientHeroStats::SpellBlock
    constexpr auto BonusSpellBlock = 0x20D8; // [SEED]; Offsets.h::HeroStats::BonusSpellBlock; via AIBaseClientHeroStats::BonusSpellBlock
    constexpr auto HPRegenRate = 0x2100; // [SEED]; Offsets.h::HeroStats::HPRegenRate; via AIBaseClientHeroStats::HPRegenRate
    constexpr auto BaseHPRegenRate = 0x2128; // [SEED]; Offsets.h::HeroStats::BaseHPRegenRate; via AIBaseClientHeroStats::BaseHPRegenRate
    constexpr auto MoveSpeed = 0x2150; // [SEED]; Offsets.h::HeroStats::MoveSpeed; via AIBaseClientHeroStats::MoveSpeed
    constexpr auto MoveSpeedBaseIncrease = 0x2178; // [SEED]; Offsets.h::HeroStats::MoveSpeedBaseIncrease; via AIBaseClientHeroStats::MoveSpeedBaseIncrease
    constexpr auto AttackRange = 0x21A0; // [SEED]; Offsets.h::HeroStats::AttackRange; via AIBaseClientHeroStats::AttackRange
    constexpr auto FlatBubbleRadiusMod = 0x21C8; // [SEED]; Offsets.h::HeroStats::FlatBubbleRadiusMod; via AIBaseClientHeroStats::FlatBubbleRadiusMod
    constexpr auto PercentBubbleRadiusMod = 0x21F0; // [SEED]; Offsets.h::HeroStats::PercentBubbleRadiusMod; via AIBaseClientHeroStats::PercentBubbleRadiusMod
    constexpr auto FlatArmorPen = 0x2218; // [SEED]; Offsets.h::HeroStats::FlatArmorPen; via AIBaseClientHeroStats::FlatArmorPen
    constexpr auto PhysicalLethality = 0x2240; // [SEED]; Offsets.h::HeroStats::PhysicalLethality; via AIBaseClientHeroStats::PhysicalLethality
    constexpr auto PercentArmorPen = 0x2268; // [SEED]; Offsets.h::HeroStats::PercentArmorPen; via AIBaseClientHeroStats::PercentArmorPen
    constexpr auto PercentBonusArmorPen = 0x2290; // [SEED]; Offsets.h::HeroStats::PercentBonusArmorPen; via AIBaseClientHeroStats::PercentBonusArmorPen
    constexpr auto PercentCritBonusArmorPen = 0x22B8; // [SEED]; Offsets.h::HeroStats::PercentCritBonusArmorPen; via AIBaseClientHeroStats::PercentCritBonusArmorPen
    constexpr auto PercentCritTotalArmorPen = 0x22E0; // [SEED]; Offsets.h::HeroStats::PercentCritTotalArmorPen; via AIBaseClientHeroStats::PercentCritTotalArmorPen
    constexpr auto FlatMagicPen = 0x2308; // [SEED]; Offsets.h::HeroStats::FlatMagicPen; via AIBaseClientHeroStats::FlatMagicPen
    constexpr auto MagicLethality = 0x2330; // [SEED]; Offsets.h::HeroStats::MagicLethality; via AIBaseClientHeroStats::MagicLethality
    constexpr auto PercentMagicPen = 0x2358; // [SEED]; Offsets.h::HeroStats::PercentMagicPen; via AIBaseClientHeroStats::PercentMagicPen
    constexpr auto PercentBonusMagicPen = 0x2380; // [SEED]; Offsets.h::HeroStats::PercentBonusMagicPen; via AIBaseClientHeroStats::PercentBonusMagicPen
    constexpr auto PercentLifeSteal = 0x23A8; // [SEED]; Offsets.h::HeroStats::PercentLifeSteal; via AIBaseClientHeroStats::PercentLifeSteal
    constexpr auto PercentSpellVamp = 0x23D0; // [SEED]; Offsets.h::HeroStats::PercentSpellVamp; via AIBaseClientHeroStats::PercentSpellVamp
    constexpr auto PercentOmnivamp = 0x23F8; // [SEED]; Offsets.h::HeroStats::PercentOmnivamp; via AIBaseClientHeroStats::PercentOmnivamp
    constexpr auto PercentPhysicalVamp = 0x2420; // [SEED]; Offsets.h::HeroStats::PercentPhysicalVamp; via AIBaseClientHeroStats::PercentPhysicalVamp
    constexpr auto PathfindingRadiusMod = 0x2448; // [SEED]; Offsets.h::HeroStats::PathfindingRadiusMod; via AIBaseClientHeroStats::PathfindingRadiusMod
    constexpr auto PercentCCReduction = 0x2470; // [SEED]; Offsets.h::HeroStats::PercentCCReduction; via AIBaseClientHeroStats::PercentCCReduction
    constexpr auto PercentEXPBonus = 0x2498; // [SEED]; Offsets.h::HeroStats::PercentEXPBonus; via AIBaseClientHeroStats::PercentEXPBonus
    constexpr auto FlatBaseArmorMod = 0x24C0; // [SEED]; Offsets.h::HeroStats::FlatBaseArmorMod; via AIBaseClientHeroStats::FlatBaseArmorMod
    constexpr auto FlatBaseSpellBlockMod = 0x24E8; // [SEED]; Offsets.h::HeroStats::FlatBaseSpellBlockMod; via AIBaseClientHeroStats::FlatBaseSpellBlockMod
    constexpr auto PARRegenRate = 0x2510; // [SEED]; Offsets.h::HeroStats::PARRegenRate; via AIBaseClientHeroStats::PARRegenRate
    constexpr auto PrimaryARBaseRegenRate = 0x2538; // [SEED]; Offsets.h::HeroStats::PrimaryARBaseRegenRate; via AIBaseClientHeroStats::PrimaryARBaseRegenRate
    constexpr auto SecondaryARRegenRate = 0x2560; // [SEED]; Offsets.h::HeroStats::SecondaryARRegenRate; via AIBaseClientHeroStats::SecondaryARRegenRate
    constexpr auto SecondaryARBaseRegenRate = 0x2588; // [SEED]; Offsets.h::HeroStats::SecondaryARBaseRegenRate; via AIBaseClientHeroStats::SecondaryARBaseRegenRate
    constexpr auto FlatBaseAttackSpeedMod = 0x25B0; // [SEED]; Offsets.h::HeroStats::FlatBaseAttackSpeedMod; via AIBaseClientHeroStats::FlatBaseAttackSpeedMod
    constexpr auto Gold = 0x2830; // [SEED]; Offsets.h::Hero::Gold; via AIBaseClientHero::Gold
    constexpr auto GoldTotal = 0x2858; // [SEED]; Offsets.h::Hero::GoldTotal; via AIBaseClientHero::GoldTotal
    constexpr auto MinimumGold = 0x2880; // [SEED]; Offsets.h::Hero::MinimumGold; via AIBaseClientHero::MinimumGold
    constexpr auto CombatType = 0x2C98; // [SEED]; Offsets.h::Hero::CombatType; via AIBaseClientHero::CombatType
    constexpr auto FollowerTargetDelay = 0x2DB8; // [SEED]; Offsets.h::Hero::FollowerTargetDelay; via AIBaseClientHero::FollowerTargetDelay
    constexpr auto Exp = 0x4CF0; // [SEED]; Offsets.h::Hero::Exp; via AIBaseClientHero::Exp
    constexpr auto LevelRef = 0x4D18; // [SEED]; Offsets.h::Hero::LevelRef; via AIBaseClientHero::LevelRef
    constexpr auto LevelUpPoints = 0x4D78; // [SEED]; Offsets.h::Hero::LevelUpPoints; via Seed.Hero::LevelUpPoints
    constexpr auto VisionScore = 0x55E0; // [SEED]; Offsets.h::Hero::VisionScore; via AIBaseClientHero::VisionScore
    constexpr auto ShutdownValue = 0x5608; // [SEED]; Offsets.h::Hero::ShutdownValue; via AIBaseClientHero::ShutdownValue
    constexpr auto BaseGoldOnDeath = 0x5630; // [SEED]; Offsets.h::Hero::BaseGoldOnDeath; via AIBaseClientHero::BaseGoldOnDeath
    constexpr auto NeutralMinionsKilled = 0x5658; // [SEED]; Offsets.h::Hero::NeutralMinionsKilled; via AIBaseClientHero::NeutralMinionsKilled
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto BuffManagerOffset = 0x28B8; // [SEED]; Offsets.h::BuffManager::Offset; via AIBaseClientBuffManager::Offset
    constexpr auto BuffEntriesEnd = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntriesEnd
    constexpr auto BuffEntryBuff = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntryBuff
    constexpr auto BuffType = 0xC; // [S]; stable layout; via AIBaseClientBuffManager::BuffType
    constexpr auto BuffNamePtr = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::BuffNamePtr
    constexpr auto BuffNameStr = 0x8; // [S]; stable layout; via AIBaseClientBuffManager::BuffNameStr
    constexpr auto BuffStartTime = 0x18; // [S]; stable layout; via AIBaseClientBuffManager::BuffStartTime
    constexpr auto BuffEndTime = 0x1C; // [S]; stable layout; via AIBaseClientBuffManager::BuffEndTime
    constexpr auto BuffStacksAlt = 0x38; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacksAlt
    constexpr auto BuffStacks = 0x78; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacks
    constexpr auto AiManagerOffset = 0x41F0; // [SEED]; Offsets.h::AiManager::Offset; via Seed.AiManager::Offset
    constexpr auto AiManagerInnerManager = 0x10; // [SEED]; Offsets.h::AiManager::InnerManager; via Seed.AiManager::InnerManager
    constexpr auto TargetPosition = 0x34; // [SEED]; Offsets.h::AiManager::TargetPosition; via Seed.AiManager::TargetPosition
    constexpr auto Velocity = 0x318; // [SEED]; Offsets.h::AiManager::Velocity; via Seed.AiManager::Velocity
    constexpr auto IsMoving = 0x31C; // [SEED]; Offsets.h::AiManager::IsMoving; via Seed.AiManager::IsMoving
    constexpr auto CurrentSegment = 0x320; // [SEED]; Offsets.h::AiManager::CurrentSegment; via Seed.AiManager::CurrentSegment
    constexpr auto PathStart = 0x330; // [SEED]; Offsets.h::AiManager::PathStart; via Seed.AiManager::PathStart
    constexpr auto PathEndFallback = 0x33C; // [SEED]; Offsets.h::AiManager::PathEnd; via Seed.AiManager::PathEnd
    constexpr auto Segments = 0x348; // [SEED]; Offsets.h::AiManager::Segments; via Seed.AiManager::Segments
    constexpr auto SegmentsCount = 0x350; // [SEED]; Offsets.h::AiManager::SegmentsCount; via Seed.AiManager::SegmentsCount
    constexpr auto DashSpeed = 0x360; // [SEED]; Offsets.h::AiManager::DashSpeed; via Seed.AiManager::DashSpeed
    constexpr auto IsDashing = 0x384; // [SEED]; Offsets.h::AiManager::IsDashing; via Seed.AiManager::IsDashing
    constexpr auto ServerPos = 0x474; // [SEED]; Offsets.h::AiManager::ServerPos; via Seed.AiManager::ServerPos
    constexpr auto MoveVec3 = 0x480; // [SEED]; Offsets.h::AiManager::MoveVec3; via Seed.AiManager::MoveVec3
} // namespace AIHeroClient

namespace AIMinionClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto MP = 0x360; // [SEED]; Offsets.h::Mana::MP; via AIBaseClientMana::MP
    constexpr auto MaxMP = 0x388; // [SEED]; Offsets.h::Mana::MaxMP; via AIBaseClientMana::MaxMP
    constexpr auto PAR = 0xE00; // [SEED]; Offsets.h::Mana::PAR; via AIBaseClientMana::PAR
    constexpr auto MaxPAR = 0xE28; // [INFERRED]; PAR + 0x28 inferred LeagueObfuscation pair for MaxPAR; via AIBaseClientMana::MaxPAR
    constexpr auto SAR = 0x108; // [SEED]; Offsets.h::Mana::SAR; via AIBaseClientMana::SAR
    constexpr auto MaxSAR = 0x130; // [SEED]; Offsets.h::Mana::MaxSAR; via AIBaseClientMana::MaxSAR
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto HeroStatsBase = 0x1B88; // [SEED]; Offsets.h::HeroStats::Base; via AIBaseClientHeroStats::Base
    constexpr auto PercentCooldownMod = 0x1B88; // [SEED]; Offsets.h::HeroStats::PercentCooldownMod; via AIBaseClientHeroStats::PercentCooldownMod
    constexpr auto AbilityHaste = 0x1BB0; // [SEED]; Offsets.h::HeroStats::AbilityHaste; via AIBaseClientHeroStats::AbilityHaste
    constexpr auto PercentCooldownCapMod = 0x1BD8; // [SEED]; Offsets.h::HeroStats::PercentCooldownCapMod; via AIBaseClientHeroStats::PercentCooldownCapMod
    constexpr auto PassiveCdEndTime = 0x1C00; // [SEED]; Offsets.h::HeroStats::PassiveCdEndTime; via AIBaseClientHeroStats::PassiveCdEndTime
    constexpr auto PassiveCdTotalTime = 0x1C28; // [SEED]; Offsets.h::HeroStats::PassiveCdTotalTime; via AIBaseClientHeroStats::PassiveCdTotalTime
    constexpr auto FlatPhysicalDmgMod = 0x1CC8; // [SEED]; Offsets.h::HeroStats::FlatPhysicalDmgMod; via AIBaseClientHeroStats::FlatPhysicalDmgMod
    constexpr auto PercentPhysicalDmgMod = 0x1CF0; // [SEED]; Offsets.h::HeroStats::PercentPhysicalDmgMod; via AIBaseClientHeroStats::PercentPhysicalDmgMod
    constexpr auto PercentBonusPhysDmgMod = 0x1D18; // [SEED]; Offsets.h::HeroStats::PercentBonusPhysDmgMod; via AIBaseClientHeroStats::PercentBonusPhysDmgMod
    constexpr auto PercentBasePhysDmgFlat = 0x1D40; // [SEED]; Offsets.h::HeroStats::PercentBasePhysDmgFlat; via AIBaseClientHeroStats::PercentBasePhysDmgFlat
    constexpr auto FlatMagicDmgMod = 0x1D68; // [SEED]; Offsets.h::HeroStats::FlatMagicDmgMod; via AIBaseClientHeroStats::FlatMagicDmgMod
    constexpr auto PercentMagicDmgMod = 0x1D90; // [SEED]; Offsets.h::HeroStats::PercentMagicDmgMod; via AIBaseClientHeroStats::PercentMagicDmgMod
    constexpr auto FlatMagicReduction = 0x1DB8; // [SEED]; Offsets.h::HeroStats::FlatMagicReduction; via AIBaseClientHeroStats::FlatMagicReduction
    constexpr auto PercentMagicReduction = 0x1DE0; // [SEED]; Offsets.h::HeroStats::PercentMagicReduction; via AIBaseClientHeroStats::PercentMagicReduction
    constexpr auto FlatCastRangeMod = 0x1E08; // [SEED]; Offsets.h::HeroStats::FlatCastRangeMod; via AIBaseClientHeroStats::FlatCastRangeMod
    constexpr auto AttackSpeedMod = 0x1E30; // [SEED]; Offsets.h::HeroStats::AttackSpeedMod; via AIBaseClientHeroStats::AttackSpeedMod
    constexpr auto PercentAttackSpeedMod = 0x1E58; // [SEED]; Offsets.h::HeroStats::PercentAttackSpeedMod; via AIBaseClientHeroStats::PercentAttackSpeedMod
    constexpr auto PercentMultiAtkSpeedMod = 0x1E80; // [SEED]; Offsets.h::HeroStats::PercentMultiAtkSpeedMod; via AIBaseClientHeroStats::PercentMultiAtkSpeedMod
    constexpr auto PercentHealingAmountMod = 0x1EA8; // [SEED]; Offsets.h::HeroStats::PercentHealingAmountMod; via AIBaseClientHeroStats::PercentHealingAmountMod
    constexpr auto BaseAttackDamage = 0x1ED0; // [SEED]; Offsets.h::HeroStats::BaseAttackDamage; via AIBaseClientHeroStats::BaseAttackDamage
    constexpr auto BaseAtkDmgSansScale = 0x1EF8; // [SEED]; Offsets.h::HeroStats::BaseAtkDmgSansScale; via AIBaseClientHeroStats::BaseAtkDmgSansScale
    constexpr auto FlatBaseAtkDmgMod = 0x1F20; // [SEED]; Offsets.h::HeroStats::FlatBaseAtkDmgMod; via AIBaseClientHeroStats::FlatBaseAtkDmgMod
    constexpr auto PercentBaseAtkDmgMod = 0x1F48; // [SEED]; Offsets.h::HeroStats::PercentBaseAtkDmgMod; via AIBaseClientHeroStats::PercentBaseAtkDmgMod
    constexpr auto BaseAbilityDamage = 0x1F70; // [SEED]; Offsets.h::HeroStats::BaseAbilityDamage; via AIBaseClientHeroStats::BaseAbilityDamage
    constexpr auto CritDamageMultiplier = 0x1F98; // [SEED]; Offsets.h::HeroStats::CritDamageMultiplier; via AIBaseClientHeroStats::CritDamageMultiplier
    constexpr auto ScaleSkinCoef = 0x1FC0; // [SEED]; Offsets.h::HeroStats::ScaleSkinCoef; via AIBaseClientHeroStats::ScaleSkinCoef
    constexpr auto Dodge = 0x1FE8; // [SEED]; Offsets.h::HeroStats::Dodge; via AIBaseClientHeroStats::Dodge
    constexpr auto Crit = 0x2010; // [SEED]; Offsets.h::HeroStats::Crit; via AIBaseClientHeroStats::Crit
    constexpr auto Armor = 0x2060; // [SEED]; Offsets.h::HeroStats::Armor; via AIBaseClientHeroStats::Armor
    constexpr auto BonusArmor = 0x2088; // [SEED]; Offsets.h::HeroStats::BonusArmor; via AIBaseClientHeroStats::BonusArmor
    constexpr auto SpellBlock = 0x20B0; // [SEED]; Offsets.h::HeroStats::SpellBlock; via AIBaseClientHeroStats::SpellBlock
    constexpr auto BonusSpellBlock = 0x20D8; // [SEED]; Offsets.h::HeroStats::BonusSpellBlock; via AIBaseClientHeroStats::BonusSpellBlock
    constexpr auto HPRegenRate = 0x2100; // [SEED]; Offsets.h::HeroStats::HPRegenRate; via AIBaseClientHeroStats::HPRegenRate
    constexpr auto BaseHPRegenRate = 0x2128; // [SEED]; Offsets.h::HeroStats::BaseHPRegenRate; via AIBaseClientHeroStats::BaseHPRegenRate
    constexpr auto MoveSpeed = 0x2150; // [SEED]; Offsets.h::HeroStats::MoveSpeed; via AIBaseClientHeroStats::MoveSpeed
    constexpr auto MoveSpeedBaseIncrease = 0x2178; // [SEED]; Offsets.h::HeroStats::MoveSpeedBaseIncrease; via AIBaseClientHeroStats::MoveSpeedBaseIncrease
    constexpr auto AttackRange = 0x21A0; // [SEED]; Offsets.h::HeroStats::AttackRange; via AIBaseClientHeroStats::AttackRange
    constexpr auto FlatBubbleRadiusMod = 0x21C8; // [SEED]; Offsets.h::HeroStats::FlatBubbleRadiusMod; via AIBaseClientHeroStats::FlatBubbleRadiusMod
    constexpr auto PercentBubbleRadiusMod = 0x21F0; // [SEED]; Offsets.h::HeroStats::PercentBubbleRadiusMod; via AIBaseClientHeroStats::PercentBubbleRadiusMod
    constexpr auto FlatArmorPen = 0x2218; // [SEED]; Offsets.h::HeroStats::FlatArmorPen; via AIBaseClientHeroStats::FlatArmorPen
    constexpr auto PhysicalLethality = 0x2240; // [SEED]; Offsets.h::HeroStats::PhysicalLethality; via AIBaseClientHeroStats::PhysicalLethality
    constexpr auto PercentArmorPen = 0x2268; // [SEED]; Offsets.h::HeroStats::PercentArmorPen; via AIBaseClientHeroStats::PercentArmorPen
    constexpr auto PercentBonusArmorPen = 0x2290; // [SEED]; Offsets.h::HeroStats::PercentBonusArmorPen; via AIBaseClientHeroStats::PercentBonusArmorPen
    constexpr auto PercentCritBonusArmorPen = 0x22B8; // [SEED]; Offsets.h::HeroStats::PercentCritBonusArmorPen; via AIBaseClientHeroStats::PercentCritBonusArmorPen
    constexpr auto PercentCritTotalArmorPen = 0x22E0; // [SEED]; Offsets.h::HeroStats::PercentCritTotalArmorPen; via AIBaseClientHeroStats::PercentCritTotalArmorPen
    constexpr auto FlatMagicPen = 0x2308; // [SEED]; Offsets.h::HeroStats::FlatMagicPen; via AIBaseClientHeroStats::FlatMagicPen
    constexpr auto MagicLethality = 0x2330; // [SEED]; Offsets.h::HeroStats::MagicLethality; via AIBaseClientHeroStats::MagicLethality
    constexpr auto PercentMagicPen = 0x2358; // [SEED]; Offsets.h::HeroStats::PercentMagicPen; via AIBaseClientHeroStats::PercentMagicPen
    constexpr auto PercentBonusMagicPen = 0x2380; // [SEED]; Offsets.h::HeroStats::PercentBonusMagicPen; via AIBaseClientHeroStats::PercentBonusMagicPen
    constexpr auto PercentLifeSteal = 0x23A8; // [SEED]; Offsets.h::HeroStats::PercentLifeSteal; via AIBaseClientHeroStats::PercentLifeSteal
    constexpr auto PercentSpellVamp = 0x23D0; // [SEED]; Offsets.h::HeroStats::PercentSpellVamp; via AIBaseClientHeroStats::PercentSpellVamp
    constexpr auto PercentOmnivamp = 0x23F8; // [SEED]; Offsets.h::HeroStats::PercentOmnivamp; via AIBaseClientHeroStats::PercentOmnivamp
    constexpr auto PercentPhysicalVamp = 0x2420; // [SEED]; Offsets.h::HeroStats::PercentPhysicalVamp; via AIBaseClientHeroStats::PercentPhysicalVamp
    constexpr auto PathfindingRadiusMod = 0x2448; // [SEED]; Offsets.h::HeroStats::PathfindingRadiusMod; via AIBaseClientHeroStats::PathfindingRadiusMod
    constexpr auto PercentCCReduction = 0x2470; // [SEED]; Offsets.h::HeroStats::PercentCCReduction; via AIBaseClientHeroStats::PercentCCReduction
    constexpr auto PercentEXPBonus = 0x2498; // [SEED]; Offsets.h::HeroStats::PercentEXPBonus; via AIBaseClientHeroStats::PercentEXPBonus
    constexpr auto FlatBaseArmorMod = 0x24C0; // [SEED]; Offsets.h::HeroStats::FlatBaseArmorMod; via AIBaseClientHeroStats::FlatBaseArmorMod
    constexpr auto FlatBaseSpellBlockMod = 0x24E8; // [SEED]; Offsets.h::HeroStats::FlatBaseSpellBlockMod; via AIBaseClientHeroStats::FlatBaseSpellBlockMod
    constexpr auto PARRegenRate = 0x2510; // [SEED]; Offsets.h::HeroStats::PARRegenRate; via AIBaseClientHeroStats::PARRegenRate
    constexpr auto PrimaryARBaseRegenRate = 0x2538; // [SEED]; Offsets.h::HeroStats::PrimaryARBaseRegenRate; via AIBaseClientHeroStats::PrimaryARBaseRegenRate
    constexpr auto SecondaryARRegenRate = 0x2560; // [SEED]; Offsets.h::HeroStats::SecondaryARRegenRate; via AIBaseClientHeroStats::SecondaryARRegenRate
    constexpr auto SecondaryARBaseRegenRate = 0x2588; // [SEED]; Offsets.h::HeroStats::SecondaryARBaseRegenRate; via AIBaseClientHeroStats::SecondaryARBaseRegenRate
    constexpr auto FlatBaseAttackSpeedMod = 0x25B0; // [SEED]; Offsets.h::HeroStats::FlatBaseAttackSpeedMod; via AIBaseClientHeroStats::FlatBaseAttackSpeedMod
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto BuffManagerOffset = 0x28B8; // [SEED]; Offsets.h::BuffManager::Offset; via AIBaseClientBuffManager::Offset
    constexpr auto BuffEntriesEnd = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntriesEnd
    constexpr auto BuffEntryBuff = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntryBuff
    constexpr auto BuffType = 0xC; // [S]; stable layout; via AIBaseClientBuffManager::BuffType
    constexpr auto BuffNamePtr = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::BuffNamePtr
    constexpr auto BuffNameStr = 0x8; // [S]; stable layout; via AIBaseClientBuffManager::BuffNameStr
    constexpr auto BuffStartTime = 0x18; // [S]; stable layout; via AIBaseClientBuffManager::BuffStartTime
    constexpr auto BuffEndTime = 0x1C; // [S]; stable layout; via AIBaseClientBuffManager::BuffEndTime
    constexpr auto BuffStacksAlt = 0x38; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacksAlt
    constexpr auto BuffStacks = 0x78; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacks
    constexpr auto AiManagerOffset = 0x41F0; // [SEED]; Offsets.h::AiManager::Offset; via Seed.AiManager::Offset
    constexpr auto AiManagerInnerManager = 0x10; // [SEED]; Offsets.h::AiManager::InnerManager; via Seed.AiManager::InnerManager
    constexpr auto TargetPosition = 0x34; // [SEED]; Offsets.h::AiManager::TargetPosition; via Seed.AiManager::TargetPosition
    constexpr auto Velocity = 0x318; // [SEED]; Offsets.h::AiManager::Velocity; via Seed.AiManager::Velocity
    constexpr auto IsMoving = 0x31C; // [SEED]; Offsets.h::AiManager::IsMoving; via Seed.AiManager::IsMoving
    constexpr auto CurrentSegment = 0x320; // [SEED]; Offsets.h::AiManager::CurrentSegment; via Seed.AiManager::CurrentSegment
    constexpr auto PathStart = 0x330; // [SEED]; Offsets.h::AiManager::PathStart; via Seed.AiManager::PathStart
    constexpr auto PathEndFallback = 0x33C; // [SEED]; Offsets.h::AiManager::PathEnd; via Seed.AiManager::PathEnd
    constexpr auto Segments = 0x348; // [SEED]; Offsets.h::AiManager::Segments; via Seed.AiManager::Segments
    constexpr auto SegmentsCount = 0x350; // [SEED]; Offsets.h::AiManager::SegmentsCount; via Seed.AiManager::SegmentsCount
    constexpr auto DashSpeed = 0x360; // [SEED]; Offsets.h::AiManager::DashSpeed; via Seed.AiManager::DashSpeed
    constexpr auto IsDashing = 0x384; // [SEED]; Offsets.h::AiManager::IsDashing; via Seed.AiManager::IsDashing
    constexpr auto ServerPos = 0x474; // [SEED]; Offsets.h::AiManager::ServerPos; via Seed.AiManager::ServerPos
    constexpr auto MoveVec3 = 0x480; // [SEED]; Offsets.h::AiManager::MoveVec3; via Seed.AiManager::MoveVec3
    constexpr auto LaneArray = 0x68; // [SEED]; Offsets.h::Minion::LaneArray; via Seed.Minion::LaneArray
    constexpr auto LaneCount = 0x70; // [SEED]; Offsets.h::Minion::LaneCount; via Seed.Minion::LaneCount
    constexpr auto LaneType = 0x4CC9; // [SEED]; Offsets.h::Minion::LaneType; via Seed.Minion::LaneType
    constexpr auto JungleTypeOffset = 0x4A84; // [SEED]; Offsets.h::JungleType::TypeOffset; via Seed.JungleType::TypeOffset
    constexpr auto TypeFlagsField = 0x4C; // [SEED]; Offsets.h::TypeFlags::ObfuscatedField; via Seed.TypeFlags::ObfuscatedField
} // namespace AIMinionClient

namespace MissileClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto SpellDataPtr = 0x128; // [SEED]; Offsets.h::Missile::SpellDataPtr; via Seed.Missile::SpellDataPtr
    constexpr auto CastInfoBase = 0x2C0; // [SEED]; Offsets.h::Missile::CastInfoBase; via Seed.Missile::CastInfoBase
    constexpr auto SpellName = 0x2E0; // [SEED]; Offsets.h::Missile::SpellName; via Seed.Missile::SpellName
    constexpr auto MissileName = 0x308; // [SEED]; Offsets.h::Missile::MissileName; via Seed.Missile::MissileName
    constexpr auto CasterNetId = 0x358; // [SEED]; Offsets.h::Missile::CasterNetId; via Seed.Missile::CasterNetId
    constexpr auto TargetNetId = 0x35C; // [SEED]; Offsets.h::Missile::TargetNetId; via Seed.Missile::TargetNetId
    constexpr auto MissileNetId = 0x364; // [SEED]; Offsets.h::Missile::MissileNetId; via Seed.Missile::MissileNetId
    constexpr auto StartPos = 0x388; // [SEED]; Offsets.h::Missile::StartPos; via Seed.Missile::StartPos
    constexpr auto EndPos = 0x394; // [SEED]; Offsets.h::Missile::EndPos; via Seed.Missile::EndPos
    constexpr auto CastEndPos = 0x3A4; // [SEED]; Offsets.h::Missile::CastEndPos; via Seed.Missile::CastEndPos
    constexpr auto SpellCastInfoSpellData = 0x0; // [SEED]; Offsets.h::SpellCastInfo::SpellData; via Raw.SpellCastInfo::SpellData
    constexpr auto SpellCastInfoSrcIndex = 0x98; // [S]; stable layout; via Raw.SpellCastInfo::SrcIndex
    constexpr auto SpellCastInfoStartPos = 0xD8; // [S]; stable layout; via Raw.SpellCastInfo::StartPos
    constexpr auto SpellCastInfoEndPos = 0xE4; // [S]; stable layout; via Raw.SpellCastInfo::EndPos
    constexpr auto SpellCastInfoCastPos = 0xF0; // [S]; stable layout; via Raw.SpellCastInfo::CastPos
    constexpr auto SpellCastInfoTargetIndex = 0x108; // [S]; stable layout; via Raw.SpellCastInfo::TargetIndex
    constexpr auto SpellCastInfoDestIndex = 0x108; // [ALIAS]; alias of TargetIndex; via Raw.SpellCastInfo::DestIndex
    constexpr auto SpellCastInfoCastDelay = 0x118; // [S]; stable layout; via Raw.SpellCastInfo::CastDelay
    constexpr auto SpellCastInfoIsAuto = 0x141; // [S]; stable layout; via Raw.SpellCastInfo::IsAuto
    constexpr auto SpellCastInfoSlot = 0x14C; // [S]; stable layout; via Raw.SpellCastInfo::Slot
} // namespace MissileClient

namespace Static {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
} // namespace Static

namespace AITurretClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace AITurretClient

namespace EffectEmitter {
    constexpr auto EmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via Seed.GameObject::EffectEmitter
    constexpr auto Data = 0x8; // [C]; chimera_structures.h::EffectEmitter::Data; via Supp.EffectEmitter::Data
    constexpr auto Attachment = 0x38; // [C]; chimera_structures.h::EffectEmitter::Attachment; via Supp.EffectEmitter::Attachment
    constexpr auto TargetAttachment = 0x48; // [C]; chimera_structures.h::EffectEmitter::TargetAttachment; via Supp.EffectEmitter::TargetAttachment
    constexpr auto AttachmentData = 0x8; // [C]; chimera_structures.h::EffectEmitterAttachment::Data; via Supp.EffectEmitterAttachment::Data
    constexpr auto AttachmentObject = 0x0; // [C]; chimera_structures.h::EffectEmitterAttachment::Object; via Supp.EffectEmitterAttachment::Object
    constexpr auto OrientationRight = 0x118; // [C]; chimera_structures.h::EffectEmitterData::OrientationRight; via Supp.EffectEmitterData::OrientationRight
    constexpr auto OrientationUp = 0x128; // [C]; chimera_structures.h::EffectEmitterData::OrientationUp; via Supp.EffectEmitterData::OrientationUp
    constexpr auto OrientationForward = 0x138; // [C]; chimera_structures.h::EffectEmitterData::OrientationForward; via Supp.EffectEmitterData::OrientationForward
} // namespace EffectEmitter

namespace BarracksDampenerClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace BarracksDampenerClient

namespace HQClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace HQClient

namespace ShopClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
} // namespace ShopClient

namespace Obj_SpawnPoint {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
} // namespace Obj_SpawnPoint
}
trankhanhtinh1 is offline

Old 22nd March 2026, 03:47 AM   #12989
caitou2024
n00bie

caitou2024's Avatar

Join Date: Apr 2024
Posts: 3
Reputation: 10
Rep Power: 49
caitou2024 has made posts that are generally average in quality
Points: 1,428, Level: 3
Points: 1,428, Level: 3 Points: 1,428, Level: 3 Points: 1,428, Level: 3
Level up: 4%, 672 Points needed
Level up: 4% Level up: 4% Level up: 4%
Activity: 4.8%
Activity: 4.8% Activity: 4.8% Activity: 4.8%
Last Achievements
League of Legends Reversal, Structs and Offsets
hello, May I ask if it is possible to direct this tool
Or I would like to know the binary code for 'Gets pellCastInfo'

Thank u!
caitou2024 is offline

Old 23rd March 2026, 06:10 AM   #12990
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by caitou2024 View Post
hello, May I ask if it is possible to direct this tool
Or I would like to know the binary code for 'Gets pellCastInfo'

Thank u!
Code:
    constexpr auto GetSpellCastInfo = 0x288D50; // [SEED]; Offsets.h::Function::GetSpellCastInfo; via Runtime.Function::GetSpellCastInfo
this is u need
trankhanhtinh1 is offline

Old 23rd March 2026, 07:08 AM   #12991
caitou2024
n00bie

caitou2024's Avatar

Join Date: Apr 2024
Posts: 3
Reputation: 10
Rep Power: 49
caitou2024 has made posts that are generally average in quality
Points: 1,428, Level: 3
Points: 1,428, Level: 3 Points: 1,428, Level: 3 Points: 1,428, Level: 3
Level up: 4%, 672 Points needed
Level up: 4% Level up: 4% Level up: 4%
Activity: 4.8%
Activity: 4.8% Activity: 4.8% Activity: 4.8%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
Code:
    constexpr auto GetSpellCastInfo = 0x288D50; // [SEED]; Offsets.h::Function::GetSpellCastInfo; via Runtime.Function::GetSpellCastInfo
this is u need
What I mean is the location code

for example 48 8B 3D ?? ?? ?? ?? FF CA
caitou2024 is offline

Old 23rd March 2026, 10:41 AM   #12992
fakekey
Member

fakekey's Avatar

Join Date: Aug 2020
Posts: 70
Reputation: 1649
Rep Power: 138
fakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Nowfakekey just Can't Stop Now
Points: 6,576, Level: 9
Points: 6,576, Level: 9 Points: 6,576, Level: 9 Points: 6,576, Level: 9
Level up: 7%, 1,024 Points needed
Level up: 7% Level up: 7% Level up: 7%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by caitou2024 View Post
What I mean is the location code

for example 48 8B 3D ?? ?? ?? ?? FF CA
open IDA, press G, input 0x288D50 (if dump file had already rebased to 0x0, otherwise: base + 0x288D50), press Enter
__________________

fakekey is offline

Old 23rd March 2026, 10:54 AM   #12993
trankhanhtinh1
Junior Member

trankhanhtinh1's Avatar

Join Date: Sep 2024
Posts: 47
Reputation: 322
Rep Power: 39
trankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a modtrankhanhtinh1 has dreams of becoming a mod
Points: 1,500, Level: 3
Points: 1,500, Level: 3 Points: 1,500, Level: 3 Points: 1,500, Level: 3
Level up: 15%, 600 Points needed
Level up: 15% Level up: 15% Level up: 15%
Activity: 9.5%
Activity: 9.5% Activity: 9.5% Activity: 9.5%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by caitou2024 View Post
What I mean is the location code

for example 48 8B 3D ?? ?? ?? ?? FF CA
use signature plugin for get it
trankhanhtinh1 is offline

Old 24th March 2026, 02:37 AM   #12994
znob
Posting Well

znob's Avatar

Join Date: Jun 2021
Posts: 34
Reputation: 150
Rep Power: 117
znob is known to create posts excellent in qualityznob is known to create posts excellent in quality
Points: 3,956, Level: 6
Points: 3,956, Level: 6 Points: 3,956, Level: 6 Points: 3,956, Level: 6
Level up: 40%, 544 Points needed
Level up: 40% Level up: 40% Level up: 40%
Activity: 7.1%
Activity: 7.1% Activity: 7.1% Activity: 7.1%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
new offset for hotfix

Code:
# pragma once

// Generated by tools/dump_object_family_offsets_ida.py
// Generated UTC: 2026-03-21T05:04:47+00:00
// Seed header: D:\source\LOL_Dumper_[unknowncheats.me]_\EnsoulSharp.SDK-master\ImGui-DirectX-11-Kiero-Hook-master\Nightsharp\core\Offsets.h
// Input file: C:\Users\MR THINH\Downloads\dump\League of Legends_exe_PID3a84_League of Legends.exe_7FF6D05B0000_x64.exe
// Module: League of Legends_exe_PID3a84_League of Legends.exe_7FF6D05B0000_x64.exe

namespace Offset {

// ----------------------------------------------------------------
// EnsoulSharp-style runtime API groups
// ----------------------------------------------------------------

namespace GameObjectsRuntime {
    constexpr auto Player = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto Objects = 0x1DA1488; // [SEED]; Offsets.h::Global::ObjectManager; via Runtime.Global::ObjectManager
    constexpr auto Heroes = 0x1DA14E0; // [SEED]; Offsets.h::Global::HeroManager; via Runtime.Global::HeroManager
    constexpr auto Minions = 0x1DA14D8; // [SEED]; Offsets.h::Global::MinionManager; via Runtime.Global::MinionManager
    constexpr auto Missiles = 0x1DA5270; // [SEED]; Offsets.h::Global::MissileManager; via Runtime.Global::MissileManager
    constexpr auto Turrets = 0x1DAE248; // [SEED]; Offsets.h::Global::TurretManager; via Runtime.Global::TurretManager
    constexpr auto UnderMouseObject = 0x19ECD78; // [SEED]; Offsets.h::Global::UnderMouseObj; via Runtime.Global::UnderMouseObj
} // namespace GameObjectsRuntime

namespace ObjectManagerRuntime {
    constexpr auto Player = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto ObjectManager = 0x1DA1488; // [SEED]; Offsets.h::Global::ObjectManager; via Runtime.Global::ObjectManager
    constexpr auto HeroManager = 0x1DA14E0; // [SEED]; Offsets.h::Global::HeroManager; via Runtime.Global::HeroManager
    constexpr auto MinionManager = 0x1DA14D8; // [SEED]; Offsets.h::Global::MinionManager; via Runtime.Global::MinionManager
    constexpr auto MissileManager = 0x1DA5270; // [SEED]; Offsets.h::Global::MissileManager; via Runtime.Global::MissileManager
    constexpr auto TurretManager = 0x1DAE248; // [SEED]; Offsets.h::Global::TurretManager; via Runtime.Global::TurretManager
    constexpr auto ManagerListItems = 0x8; // [S]; stable layout; via Runtime.ManagerList::Items
    constexpr auto ManagerListSize = 0x10; // [S]; stable layout; via Runtime.ManagerList::Size
    constexpr auto GetFirstObject = 0x9C39B0; // [SEED]; Offsets.h::Function::GetFirstObject; via Runtime.Function::GetFirstObject
    constexpr auto GetFirstObjectAlt = 0x9C39B0; // [SEED]; Offsets.h::Function::GetFirstObjectAlt; via Runtime.Function::GetFirstObjectAlt
    constexpr auto GetNextObject = 0x523760; // [SEED]; Offsets.h::Function::GetNextObject; via Runtime.Function::GetNextObject
    constexpr auto FindObject = 0x522530; // [SEED]; Offsets.h::Function::FindObject; via Runtime.Function::FindObject
} // namespace ObjectManagerRuntime

namespace GameRuntime {
    constexpr auto LocalPlayer = 0x1DD33B0; // [SEED]; Offsets.h::Global::LocalPlayer; via Runtime.Global::LocalPlayer
    constexpr auto GameTime = 0x1DAF720; // [SEED]; Offsets.h::Global::GameTime; via Runtime.Global::GameTime
    constexpr auto NetInstance = 0x1DA1480; // [SEED]; Offsets.h::Global::NetInstance; via Runtime.Global::NetInstance
    constexpr auto ChatClient = 0x1DB43E0; // [SEED]; Offsets.h::Global::ChatClient; via Runtime.Global::ChatClient
    constexpr auto ChatInstance = 0x1DA5480; // [SEED]; Offsets.h::Global::ChatInstance; via Runtime.Global::ChatInstance
    constexpr auto ShopInstance = 0x1DB43F8; // [SEED]; Offsets.h::Global::ShopInstance; via Runtime.Global::ShopInstance
    constexpr auto OpenWindowsArray = 0x1E66E78; // [SEED]; Offsets.h::Global::OpenWindowsArray; via Runtime.Global::OpenWindowsArray
    constexpr auto OpenWindowsCount = 0x1E66E80; // [SEED]; Offsets.h::Global::OpenWindowsCount; via Runtime.Global::OpenWindowsCount
    constexpr auto CursorPosRaw = 0x1E2DC38; // [SEED]; Offsets.h::Global::CursorInstance; via Runtime.Global::CursorInstance
    constexpr auto MouseScreenVec2 = 0x1DA5218; // [SEED]; Offsets.h::Global::MouseScreenVec2; via Runtime.Global::MouseScreenVec2
    constexpr auto UnderMouseObject = 0x19ECD78; // [SEED]; Offsets.h::Global::UnderMouseObj; via Runtime.Global::UnderMouseObj
    constexpr auto GetPing = 0x677420; // [SEED]; Offsets.h::Function::GetPing; via Runtime.Function::GetPing
    constexpr auto GetMapID = 0x2933B0; // [SEED]; Offsets.h::Function::GetMapID; via Runtime.Function::GetMapID
    constexpr auto PrintChat = 0x10B11B0; // [SEED]; Offsets.h::Function::PrintChat; via Runtime.Function::PrintChat
} // namespace GameRuntime

namespace DrawingRuntime {
    constexpr auto WorldToScreen = 0x1260DC0; // [SEED]; Offsets.h::Function::WorldToScreen; via Runtime.Function::WorldToScreen
    constexpr auto HudInstance = 0x1DA1628; // [SEED]; Offsets.h::Global::HudInstance; via Runtime.Global::HudInstance
    constexpr auto ViewPort = 0x1DB4398; // [SEED]; Offsets.h::Global::ViewPort; via Runtime.Global::ViewPort
    constexpr auto ViewPort2 = 0x1E68458; // [SEED]; Offsets.h::Global::ViewPort2; via Runtime.Global::ViewPort2
    constexpr auto Renderer = 0x1E68450; // [SEED]; Offsets.h::Global::r3dRenderer; via Runtime.Global::r3dRenderer
    constexpr auto Camera = 0x18; // [SEED]; Offsets.h::Hud::Camera; via Runtime.Hud::Camera
    constexpr auto Input = 0x28; // [SEED]; Offsets.h::Hud::Input; via Runtime.Hud::Input
    constexpr auto UserData = 0x60; // [SEED]; Offsets.h::Hud::UserData; via Runtime.Hud::UserData
    constexpr auto SpellInfo = 0x68; // [SEED]; Offsets.h::Hud::SpellInfo; via Runtime.Hud::SpellInfo
    constexpr auto CameraZoom = 0x324; // [SEED]; Offsets.h::Hud::CameraZoom; via Runtime.Hud::CameraZoom
    constexpr auto CameraZoomLimits = 0x310; // [SEED]; Offsets.h::Hud::CameraZoomLimits; via Runtime.Hud::CameraZoomLimits
    constexpr auto ZoomLimitsMin = 0x24; // [SEED]; Offsets.h::Hud::ZoomLimitsMin; via Runtime.Hud::ZoomLimitsMin
    constexpr auto ZoomLimitsMax = 0x28; // [SEED]; Offsets.h::Hud::ZoomLimitsMax; via Runtime.Hud::ZoomLimitsMax
    constexpr auto AltZoomLimits = 0x3D0; // [SEED]; Offsets.h::Hud::AltZoomLimits; via Runtime.Hud::AltZoomLimits
    constexpr auto ZoomLockFlag1 = 0x344; // [SEED]; Offsets.h::Hud::ZoomLockFlag1; via Runtime.Hud::ZoomLockFlag1
    constexpr auto ZoomLockFlag2 = 0x345; // [SEED]; Offsets.h::Hud::ZoomLockFlag2; via Runtime.Hud::ZoomLockFlag2
    constexpr auto MouseWorldPos = 0x34; // [SEED]; Offsets.h::Hud::MouseWorldPos; via Runtime.Hud::MouseWorldPos
    constexpr auto SelectedObjNetId = 0x28; // [SEED]; Offsets.h::Hud::SelectedObjNetId; via Runtime.Hud::SelectedObjNetId
    constexpr auto ChatOpen = 0x10; // [SEED]; Offsets.h::Hud::ChatOpen; via Runtime.Hud::ChatOpen
    constexpr auto ViewportW2S = 0x2B0; // [SEED]; Offsets.h::Hud::ViewportW2S; via Runtime.Hud::ViewportW2S
} // namespace DrawingRuntime

namespace ControlRuntime {
    constexpr auto IssueOrder = 0x2A5040; // [SEED]; Offsets.h::Function::IssueOrder -> IssueOrderCore; via Runtime.Function::IssueOrder
    constexpr auto IssueOrderCore = 0x2A5040; // [SEED]; Offsets.h::Function::IssueOrderCore; via Runtime.Function::IssueOrderCore
    constexpr auto CastSpellSafe = 0xBB8950; // [SEED]; Offsets.h::Function::CastSpellSafe; via Runtime.Function::CastSpellSafe
    constexpr auto GetSpellCastInfo = 0x288D50; // [SEED]; Offsets.h::Function::GetSpellCastInfo; via Runtime.Function::GetSpellCastInfo
    constexpr auto GetSpellSlot = 0x905BC0; // [SEED]; Offsets.h::Function::GetSpellSlot; via Runtime.Function::GetSpellSlot
    constexpr auto GetResourceType = 0x286070; // [SEED]; Offsets.h::Function::GetResourceType; via Runtime.Function::GetResourceType
    constexpr auto GetAttackDelay = 0x53A3C0; // [SEED]; Offsets.h::Function::GetAttackDelay; via Runtime.Function::GetAttackDelay
    constexpr auto GetAttackWindup = 0x53A2C0; // [SEED]; Offsets.h::Function::GetAttackWindup; via Runtime.Function::GetAttackWindup
    constexpr auto GetBoundingRadius = 0x28A600; // [SEED]; Offsets.h::Function::GetBoundingRadius; via Runtime.Function::GetBoundingRadius
    constexpr auto IssueOrderFlag = 0x1D04FA8; // [SEED]; Offsets.h::Flag::IssueOrderFlag; via Runtime.Flag::IssueOrderFlag
    constexpr auto CastSpellFlag = 0x1D04F40; // [SEED]; Offsets.h::Flag::CastSpellFlag; via Runtime.Flag::CastSpellFlag
} // namespace ControlRuntime

namespace EventRuntime {
    constexpr auto CreateClientEffect = 0x83C170; // [SEED]; Offsets.h::Function::CreateClientEffect; via Runtime.Function::CreateClientEffect
    constexpr auto OnCreateObject = 0x527930; // [SEED]; Offsets.h::Function::OnCreateObject; via Runtime.Function::OnCreateObject
    constexpr auto OnGameUpdate = 0x5215E0; // [SEED]; Offsets.h::Function::OnGameUpdate; via Runtime.Function::OnGameUpdate
    constexpr auto OnProcessSpell = 0x91D1B0; // [SEED]; Offsets.h::Function::OnProcessSpell; via Runtime.Function::OnProcessSpell
    constexpr auto OnSpellImpact = 0x914320; // [SEED]; Offsets.h::Function::OnSpellImpact; via Runtime.Function::OnSpellImpact
    constexpr auto OnStopCast = 0x91D750; // [SEED]; Offsets.h::Function::OnStopCast; via Runtime.Function::OnStopCast
    constexpr auto OnFinishCast = 0x2CBE30; // [SEED]; Offsets.h::Function::OnFinishCast; via Runtime.Function::OnFinishCast
    constexpr auto OnBuffAdd = 0xBD0B40; // [SEED]; Offsets.h::Function::OnBuffAdd; via Runtime.Function::OnBuffAdd
} // namespace EventRuntime

namespace NavGridRuntime {
    constexpr auto NavGrid = 0x1DA51E0; // [SEED]; Offsets.h::Global::NavGrid; via Runtime.Global::NavGrid
    constexpr auto GetCollisionFlags = 0x11B29D0; // [SEED]; Offsets.h::Function::GetCollisionFlags; via Runtime.Function::GetCollisionFlags
    constexpr auto GetAiManager = 0x292420; // [SEED]; Offsets.h::Function::GetAiManager; via Runtime.Function::GetAiManager
    constexpr auto GetAiManagerInner = 0x293A10; // [SEED]; Offsets.h::Function::GetAiManagerInner; via Runtime.Function::GetAiManagerInner
    constexpr auto NavGridMgr = 0x8; // [SEED]; Offsets.h::NavGrid::NavGridMgr; via Runtime.NavGrid::NavGridMgr
    constexpr auto MinX = 0xEC; // [SEED]; Offsets.h::NavGrid::MinX; via Runtime.NavGrid::MinX
    constexpr auto MinZ = 0xF4; // [SEED]; Offsets.h::NavGrid::MinZ; via Runtime.NavGrid::MinZ
    constexpr auto MaxX = 0xF8; // [SEED]; Offsets.h::NavGrid::MaxX; via Runtime.NavGrid::MaxX
    constexpr auto MaxZ = 0x100; // [SEED]; Offsets.h::NavGrid::MaxZ; via Runtime.NavGrid::MaxZ
    constexpr auto Data = 0x110; // [SEED]; Offsets.h::NavGrid::Data; via Runtime.NavGrid::Data
    constexpr auto Width = 0x708; // [SEED]; Offsets.h::NavGrid::Width; via Runtime.NavGrid::Width
    constexpr auto Height = 0x70C; // [SEED]; Offsets.h::NavGrid::Height; via Runtime.NavGrid::Height
    constexpr auto Scale = 0x710; // [SEED]; Offsets.h::NavGrid::Scale; via Runtime.NavGrid::Scale
    constexpr auto InverseScale = 0x714; // [SEED]; Offsets.h::NavGrid::InverseScale; via Runtime.NavGrid::InverseScale
    constexpr auto GrassRegions = 0x158; // [SEED]; Offsets.h::NavGrid::GrassRegions; via Runtime.NavGrid::GrassRegions
    constexpr auto CellSize = 0x10; // [SEED]; Offsets.h::NavGrid::CellSize; via Runtime.NavGrid::CellSize
    constexpr auto FlagWall = 0x1; // [SEED]; Offsets.h::NavGrid::FLAG_WALL; via Runtime.NavGrid::FLAG_WALL
    constexpr auto FlagNoWalk = 0x2; // [SEED]; Offsets.h::NavGrid::FLAG_NOWALK; via Runtime.NavGrid::FLAG_NOWALK
    constexpr auto FlagBrush = 0xC00; // [SEED]; Offsets.h::NavGrid::FLAG_BRUSH; via Runtime.NavGrid::FLAG_BRUSH
    constexpr auto FlagSpecial = 0x1000; // [SEED]; Offsets.h::NavGrid::FLAG_SPECIAL; via Runtime.NavGrid::FLAG_SPECIAL
} // namespace NavGridRuntime

namespace SpellRuntime {
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto SpellCastInfoSpellData = 0x0; // [SEED]; Offsets.h::SpellCastInfo::SpellData; via Runtime.SpellCastInfo::SpellData
    constexpr auto SpellCastInfoSrcIndex = 0x98; // [S]; stable layout; via Runtime.SpellCastInfo::SrcIndex
    constexpr auto SpellCastInfoStartPos = 0xD8; // [S]; stable layout; via Runtime.SpellCastInfo::StartPos
    constexpr auto SpellCastInfoEndPos = 0xE4; // [S]; stable layout; via Runtime.SpellCastInfo::EndPos
    constexpr auto SpellCastInfoCastPos = 0xF0; // [S]; stable layout; via Runtime.SpellCastInfo::CastPos
    constexpr auto SpellCastInfoTargetIndex = 0x108; // [S]; stable layout; via Runtime.SpellCastInfo::TargetIndex
    constexpr auto SpellCastInfoDestIndex = 0x108; // [ALIAS]; alias of TargetIndex; via Runtime.SpellCastInfo::DestIndex
    constexpr auto SpellCastInfoCastDelay = 0x118; // [S]; stable layout; via Runtime.SpellCastInfo::CastDelay
    constexpr auto SpellCastInfoIsSpell = 0x134; // [SEED]; Offsets.h::SpellCastInfo::IsSpell; via Runtime.SpellCastInfo::IsSpell
    constexpr auto SpellCastInfoIsSpecialAttack = 0x13E; // [SEED]; Offsets.h::SpellCastInfo::IsSpecialAttack; via Runtime.SpellCastInfo::IsSpecialAttack
    constexpr auto SpellCastInfoIsAuto = 0x141; // [S]; stable layout; via Runtime.SpellCastInfo::IsAuto
    constexpr auto SpellCastInfoSlot = 0x14C; // [S]; stable layout; via Runtime.SpellCastInfo::Slot
} // namespace SpellRuntime

namespace ItemRuntime {
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
    constexpr auto SlotInfo = 0x10; // [S]; stable layout; via Runtime.ItemSystem::SlotInfo
    constexpr auto InfoData = 0x38; // [S]; stable layout; via Runtime.ItemSystem::InfoData
    constexpr auto InfoStacks = 0x64; // [S]; stable layout; via Runtime.ItemSystem::InfoStacks
    constexpr auto DataItemId = 0xB4; // [SEED]; Offsets.h::ItemSystem::DataItemId; via Runtime.ItemSystem::DataItemId
    constexpr auto DataAbilityHaste = 0x160; // [SEED]; Offsets.h::ItemSystem::DataAbilityHaste; via Runtime.ItemSystem::DataAbilityHaste
    constexpr auto DataHealth = 0x164; // [SEED]; Offsets.h::ItemSystem::DataHealth; via Runtime.ItemSystem::DataHealth
    constexpr auto DataArmor = 0x19C; // [SEED]; Offsets.h::ItemSystem::DataArmor; via Runtime.ItemSystem::DataArmor
    constexpr auto DataMR = 0x1BC; // [SEED]; Offsets.h::ItemSystem::DataMR; via Runtime.ItemSystem::DataMR
    constexpr auto DataAD = 0x1D8; // [SEED]; Offsets.h::ItemSystem::DataAD; via Runtime.ItemSystem::DataAD
    constexpr auto DataAP = 0x1E0; // [SEED]; Offsets.h::ItemSystem::DataAP; via Runtime.ItemSystem::DataAP
    constexpr auto DataAtkSpeedMult = 0x20C; // [SEED]; Offsets.h::ItemSystem::DataAtkSpeedMult; via Runtime.ItemSystem::DataAtkSpeedMult
} // namespace ItemRuntime

namespace ClassificationRuntime {
    constexpr auto TypeFlagsField = 0x4C; // [SEED]; Offsets.h::TypeFlags::ObfuscatedField; via Runtime.TypeFlags::ObfuscatedField
    constexpr auto TypeIsObjectAI = 0x400; // [SEED]; Offsets.h::TypeFlags::IsObjectAI; via Runtime.TypeFlags::IsObjectAI
    constexpr auto TypeMinion = 0x800; // [SEED]; Offsets.h::TypeFlags::Minion; via Runtime.TypeFlags::Minion
    constexpr auto TypeHero = 0x1000; // [SEED]; Offsets.h::TypeFlags::Hero; via Runtime.TypeFlags::Hero
    constexpr auto TypeTurret = 0x2000; // [SEED]; Offsets.h::TypeFlags::Turret; via Runtime.TypeFlags::Turret
    constexpr auto TypePlant = 0x8000; // [SEED]; Offsets.h::TypeFlags::Plant; via Runtime.TypeFlags::Plant
    constexpr auto TypeLargeMonster = 0x80; // [SEED]; Offsets.h::TypeFlags::LargeMonster; via Runtime.TypeFlags::LargeMonster
    constexpr auto TypeBuffMonster = 0x100; // [SEED]; Offsets.h::TypeFlags::BuffMonster; via Runtime.TypeFlags::BuffMonster
    constexpr auto TypeMinionSummon = 0x100; // [SEED]; Offsets.h::TypeFlags::MinionSummon; via Runtime.TypeFlags::MinionSummon
    constexpr auto TypeAttackableObj = 0x8; // [SEED]; Offsets.h::TypeFlags::AttackableObj; via Runtime.TypeFlags::AttackableObj
    constexpr auto TypeVisibleObj = 0x10; // [SEED]; Offsets.h::TypeFlags::VisibleObj; via Runtime.TypeFlags::VisibleObj
    constexpr auto TypeRenderTarget = 0x20; // [SEED]; Offsets.h::TypeFlags::RenderTarget; via Runtime.TypeFlags::RenderTarget
    constexpr auto TypeIsRecalling = 0x4000; // [SEED]; Offsets.h::TypeFlags::IsRecalling; via Runtime.TypeFlags::IsRecalling
    constexpr auto MinionLaneArray = 0x68; // [SEED]; Offsets.h::Minion::LaneArray; via Runtime.Minion::LaneArray
    constexpr auto MinionLaneCount = 0x70; // [SEED]; Offsets.h::Minion::LaneCount; via Runtime.Minion::LaneCount
    constexpr auto MinionLaneType = 0x4CC9; // [SEED]; Offsets.h::Minion::LaneType; via Runtime.Minion::LaneType
    constexpr auto MinionClassUnset = 0x0; // [SEED]; Offsets.h::MinionClass::Unset; via Runtime.MinionClass::Unset
    constexpr auto MinionClassPet = 0x1; // [SEED]; Offsets.h::MinionClass::Pet; via Runtime.MinionClass::Pet
    constexpr auto MinionClassJungleMonster = 0x2; // [SEED]; Offsets.h::MinionClass::JungleMonster; via Runtime.MinionClass::JungleMonster
    constexpr auto MinionClassTeamMinion = 0x3; // [SEED]; Offsets.h::MinionClass::TeamMinion; via Runtime.MinionClass::TeamMinion
    constexpr auto MinionClassMeleeLaneMinion = 0x4; // [SEED]; Offsets.h::MinionClass::MeleeLaneMinion; via Runtime.MinionClass::MeleeLaneMinion
    constexpr auto MinionClassRangedLaneMinion = 0x5; // [SEED]; Offsets.h::MinionClass::RangedLaneMinion; via Runtime.MinionClass::RangedLaneMinion
    constexpr auto MinionClassSiegeLaneMinion = 0x6; // [SEED]; Offsets.h::MinionClass::SiegeLaneMinion; via Runtime.MinionClass::SiegeLaneMinion
    constexpr auto MinionClassSuperLaneMinion = 0x7; // [SEED]; Offsets.h::MinionClass::SuperLaneMinion; via Runtime.MinionClass::SuperLaneMinion
    constexpr auto JungleTypeOffset = 0x4A84; // [SEED]; Offsets.h::JungleType::TypeOffset; via Runtime.JungleType::TypeOffset
    constexpr auto JungleTypeNormal = 0x0; // [SEED]; Offsets.h::JungleType::Normal; via Runtime.JungleType::Normal
    constexpr auto JungleTypeBaron = 0x1; // [SEED]; Offsets.h::JungleType::Baron; via Runtime.JungleType::Baron
    constexpr auto JungleTypeDragon = 0x2; // [SEED]; Offsets.h::JungleType::Dragon; via Runtime.JungleType::Dragon
} // namespace ClassificationRuntime

// ----------------------------------------------------------------
// EnsoulSharp-style object family groups
// ----------------------------------------------------------------

namespace All {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
} // namespace All

namespace AttackableUnit {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
} // namespace AttackableUnit

namespace AIHeroClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto MP = 0x360; // [SEED]; Offsets.h::Mana::MP; via AIBaseClientMana::MP
    constexpr auto MaxMP = 0x388; // [SEED]; Offsets.h::Mana::MaxMP; via AIBaseClientMana::MaxMP
    constexpr auto PAR = 0xE00; // [SEED]; Offsets.h::Mana::PAR; via AIBaseClientMana::PAR
    constexpr auto MaxPAR = 0xE28; // [INFERRED]; PAR + 0x28 inferred LeagueObfuscation pair for MaxPAR; via AIBaseClientMana::MaxPAR
    constexpr auto SAR = 0x108; // [SEED]; Offsets.h::Mana::SAR; via AIBaseClientMana::SAR
    constexpr auto MaxSAR = 0x130; // [SEED]; Offsets.h::Mana::MaxSAR; via AIBaseClientMana::MaxSAR
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto HeroStatsBase = 0x1B88; // [SEED]; Offsets.h::HeroStats::Base; via AIBaseClientHeroStats::Base
    constexpr auto PercentCooldownMod = 0x1B88; // [SEED]; Offsets.h::HeroStats::PercentCooldownMod; via AIBaseClientHeroStats::PercentCooldownMod
    constexpr auto AbilityHaste = 0x1BB0; // [SEED]; Offsets.h::HeroStats::AbilityHaste; via AIBaseClientHeroStats::AbilityHaste
    constexpr auto PercentCooldownCapMod = 0x1BD8; // [SEED]; Offsets.h::HeroStats::PercentCooldownCapMod; via AIBaseClientHeroStats::PercentCooldownCapMod
    constexpr auto PassiveCdEndTime = 0x1C00; // [SEED]; Offsets.h::HeroStats::PassiveCdEndTime; via AIBaseClientHeroStats::PassiveCdEndTime
    constexpr auto PassiveCdTotalTime = 0x1C28; // [SEED]; Offsets.h::HeroStats::PassiveCdTotalTime; via AIBaseClientHeroStats::PassiveCdTotalTime
    constexpr auto FlatPhysicalDmgMod = 0x1CC8; // [SEED]; Offsets.h::HeroStats::FlatPhysicalDmgMod; via AIBaseClientHeroStats::FlatPhysicalDmgMod
    constexpr auto PercentPhysicalDmgMod = 0x1CF0; // [SEED]; Offsets.h::HeroStats::PercentPhysicalDmgMod; via AIBaseClientHeroStats::PercentPhysicalDmgMod
    constexpr auto PercentBonusPhysDmgMod = 0x1D18; // [SEED]; Offsets.h::HeroStats::PercentBonusPhysDmgMod; via AIBaseClientHeroStats::PercentBonusPhysDmgMod
    constexpr auto PercentBasePhysDmgFlat = 0x1D40; // [SEED]; Offsets.h::HeroStats::PercentBasePhysDmgFlat; via AIBaseClientHeroStats::PercentBasePhysDmgFlat
    constexpr auto FlatMagicDmgMod = 0x1D68; // [SEED]; Offsets.h::HeroStats::FlatMagicDmgMod; via AIBaseClientHeroStats::FlatMagicDmgMod
    constexpr auto PercentMagicDmgMod = 0x1D90; // [SEED]; Offsets.h::HeroStats::PercentMagicDmgMod; via AIBaseClientHeroStats::PercentMagicDmgMod
    constexpr auto FlatMagicReduction = 0x1DB8; // [SEED]; Offsets.h::HeroStats::FlatMagicReduction; via AIBaseClientHeroStats::FlatMagicReduction
    constexpr auto PercentMagicReduction = 0x1DE0; // [SEED]; Offsets.h::HeroStats::PercentMagicReduction; via AIBaseClientHeroStats::PercentMagicReduction
    constexpr auto FlatCastRangeMod = 0x1E08; // [SEED]; Offsets.h::HeroStats::FlatCastRangeMod; via AIBaseClientHeroStats::FlatCastRangeMod
    constexpr auto AttackSpeedMod = 0x1E30; // [SEED]; Offsets.h::HeroStats::AttackSpeedMod; via AIBaseClientHeroStats::AttackSpeedMod
    constexpr auto PercentAttackSpeedMod = 0x1E58; // [SEED]; Offsets.h::HeroStats::PercentAttackSpeedMod; via AIBaseClientHeroStats::PercentAttackSpeedMod
    constexpr auto PercentMultiAtkSpeedMod = 0x1E80; // [SEED]; Offsets.h::HeroStats::PercentMultiAtkSpeedMod; via AIBaseClientHeroStats::PercentMultiAtkSpeedMod
    constexpr auto PercentHealingAmountMod = 0x1EA8; // [SEED]; Offsets.h::HeroStats::PercentHealingAmountMod; via AIBaseClientHeroStats::PercentHealingAmountMod
    constexpr auto BaseAttackDamage = 0x1ED0; // [SEED]; Offsets.h::HeroStats::BaseAttackDamage; via AIBaseClientHeroStats::BaseAttackDamage
    constexpr auto BaseAtkDmgSansScale = 0x1EF8; // [SEED]; Offsets.h::HeroStats::BaseAtkDmgSansScale; via AIBaseClientHeroStats::BaseAtkDmgSansScale
    constexpr auto FlatBaseAtkDmgMod = 0x1F20; // [SEED]; Offsets.h::HeroStats::FlatBaseAtkDmgMod; via AIBaseClientHeroStats::FlatBaseAtkDmgMod
    constexpr auto PercentBaseAtkDmgMod = 0x1F48; // [SEED]; Offsets.h::HeroStats::PercentBaseAtkDmgMod; via AIBaseClientHeroStats::PercentBaseAtkDmgMod
    constexpr auto BaseAbilityDamage = 0x1F70; // [SEED]; Offsets.h::HeroStats::BaseAbilityDamage; via AIBaseClientHeroStats::BaseAbilityDamage
    constexpr auto CritDamageMultiplier = 0x1F98; // [SEED]; Offsets.h::HeroStats::CritDamageMultiplier; via AIBaseClientHeroStats::CritDamageMultiplier
    constexpr auto ScaleSkinCoef = 0x1FC0; // [SEED]; Offsets.h::HeroStats::ScaleSkinCoef; via AIBaseClientHeroStats::ScaleSkinCoef
    constexpr auto Dodge = 0x1FE8; // [SEED]; Offsets.h::HeroStats::Dodge; via AIBaseClientHeroStats::Dodge
    constexpr auto Crit = 0x2010; // [SEED]; Offsets.h::HeroStats::Crit; via AIBaseClientHeroStats::Crit
    constexpr auto Armor = 0x2060; // [SEED]; Offsets.h::HeroStats::Armor; via AIBaseClientHeroStats::Armor
    constexpr auto BonusArmor = 0x2088; // [SEED]; Offsets.h::HeroStats::BonusArmor; via AIBaseClientHeroStats::BonusArmor
    constexpr auto SpellBlock = 0x20B0; // [SEED]; Offsets.h::HeroStats::SpellBlock; via AIBaseClientHeroStats::SpellBlock
    constexpr auto BonusSpellBlock = 0x20D8; // [SEED]; Offsets.h::HeroStats::BonusSpellBlock; via AIBaseClientHeroStats::BonusSpellBlock
    constexpr auto HPRegenRate = 0x2100; // [SEED]; Offsets.h::HeroStats::HPRegenRate; via AIBaseClientHeroStats::HPRegenRate
    constexpr auto BaseHPRegenRate = 0x2128; // [SEED]; Offsets.h::HeroStats::BaseHPRegenRate; via AIBaseClientHeroStats::BaseHPRegenRate
    constexpr auto MoveSpeed = 0x2150; // [SEED]; Offsets.h::HeroStats::MoveSpeed; via AIBaseClientHeroStats::MoveSpeed
    constexpr auto MoveSpeedBaseIncrease = 0x2178; // [SEED]; Offsets.h::HeroStats::MoveSpeedBaseIncrease; via AIBaseClientHeroStats::MoveSpeedBaseIncrease
    constexpr auto AttackRange = 0x21A0; // [SEED]; Offsets.h::HeroStats::AttackRange; via AIBaseClientHeroStats::AttackRange
    constexpr auto FlatBubbleRadiusMod = 0x21C8; // [SEED]; Offsets.h::HeroStats::FlatBubbleRadiusMod; via AIBaseClientHeroStats::FlatBubbleRadiusMod
    constexpr auto PercentBubbleRadiusMod = 0x21F0; // [SEED]; Offsets.h::HeroStats::PercentBubbleRadiusMod; via AIBaseClientHeroStats::PercentBubbleRadiusMod
    constexpr auto FlatArmorPen = 0x2218; // [SEED]; Offsets.h::HeroStats::FlatArmorPen; via AIBaseClientHeroStats::FlatArmorPen
    constexpr auto PhysicalLethality = 0x2240; // [SEED]; Offsets.h::HeroStats::PhysicalLethality; via AIBaseClientHeroStats::PhysicalLethality
    constexpr auto PercentArmorPen = 0x2268; // [SEED]; Offsets.h::HeroStats::PercentArmorPen; via AIBaseClientHeroStats::PercentArmorPen
    constexpr auto PercentBonusArmorPen = 0x2290; // [SEED]; Offsets.h::HeroStats::PercentBonusArmorPen; via AIBaseClientHeroStats::PercentBonusArmorPen
    constexpr auto PercentCritBonusArmorPen = 0x22B8; // [SEED]; Offsets.h::HeroStats::PercentCritBonusArmorPen; via AIBaseClientHeroStats::PercentCritBonusArmorPen
    constexpr auto PercentCritTotalArmorPen = 0x22E0; // [SEED]; Offsets.h::HeroStats::PercentCritTotalArmorPen; via AIBaseClientHeroStats::PercentCritTotalArmorPen
    constexpr auto FlatMagicPen = 0x2308; // [SEED]; Offsets.h::HeroStats::FlatMagicPen; via AIBaseClientHeroStats::FlatMagicPen
    constexpr auto MagicLethality = 0x2330; // [SEED]; Offsets.h::HeroStats::MagicLethality; via AIBaseClientHeroStats::MagicLethality
    constexpr auto PercentMagicPen = 0x2358; // [SEED]; Offsets.h::HeroStats::PercentMagicPen; via AIBaseClientHeroStats::PercentMagicPen
    constexpr auto PercentBonusMagicPen = 0x2380; // [SEED]; Offsets.h::HeroStats::PercentBonusMagicPen; via AIBaseClientHeroStats::PercentBonusMagicPen
    constexpr auto PercentLifeSteal = 0x23A8; // [SEED]; Offsets.h::HeroStats::PercentLifeSteal; via AIBaseClientHeroStats::PercentLifeSteal
    constexpr auto PercentSpellVamp = 0x23D0; // [SEED]; Offsets.h::HeroStats::PercentSpellVamp; via AIBaseClientHeroStats::PercentSpellVamp
    constexpr auto PercentOmnivamp = 0x23F8; // [SEED]; Offsets.h::HeroStats::PercentOmnivamp; via AIBaseClientHeroStats::PercentOmnivamp
    constexpr auto PercentPhysicalVamp = 0x2420; // [SEED]; Offsets.h::HeroStats::PercentPhysicalVamp; via AIBaseClientHeroStats::PercentPhysicalVamp
    constexpr auto PathfindingRadiusMod = 0x2448; // [SEED]; Offsets.h::HeroStats::PathfindingRadiusMod; via AIBaseClientHeroStats::PathfindingRadiusMod
    constexpr auto PercentCCReduction = 0x2470; // [SEED]; Offsets.h::HeroStats::PercentCCReduction; via AIBaseClientHeroStats::PercentCCReduction
    constexpr auto PercentEXPBonus = 0x2498; // [SEED]; Offsets.h::HeroStats::PercentEXPBonus; via AIBaseClientHeroStats::PercentEXPBonus
    constexpr auto FlatBaseArmorMod = 0x24C0; // [SEED]; Offsets.h::HeroStats::FlatBaseArmorMod; via AIBaseClientHeroStats::FlatBaseArmorMod
    constexpr auto FlatBaseSpellBlockMod = 0x24E8; // [SEED]; Offsets.h::HeroStats::FlatBaseSpellBlockMod; via AIBaseClientHeroStats::FlatBaseSpellBlockMod
    constexpr auto PARRegenRate = 0x2510; // [SEED]; Offsets.h::HeroStats::PARRegenRate; via AIBaseClientHeroStats::PARRegenRate
    constexpr auto PrimaryARBaseRegenRate = 0x2538; // [SEED]; Offsets.h::HeroStats::PrimaryARBaseRegenRate; via AIBaseClientHeroStats::PrimaryARBaseRegenRate
    constexpr auto SecondaryARRegenRate = 0x2560; // [SEED]; Offsets.h::HeroStats::SecondaryARRegenRate; via AIBaseClientHeroStats::SecondaryARRegenRate
    constexpr auto SecondaryARBaseRegenRate = 0x2588; // [SEED]; Offsets.h::HeroStats::SecondaryARBaseRegenRate; via AIBaseClientHeroStats::SecondaryARBaseRegenRate
    constexpr auto FlatBaseAttackSpeedMod = 0x25B0; // [SEED]; Offsets.h::HeroStats::FlatBaseAttackSpeedMod; via AIBaseClientHeroStats::FlatBaseAttackSpeedMod
    constexpr auto Gold = 0x2830; // [SEED]; Offsets.h::Hero::Gold; via AIBaseClientHero::Gold
    constexpr auto GoldTotal = 0x2858; // [SEED]; Offsets.h::Hero::GoldTotal; via AIBaseClientHero::GoldTotal
    constexpr auto MinimumGold = 0x2880; // [SEED]; Offsets.h::Hero::MinimumGold; via AIBaseClientHero::MinimumGold
    constexpr auto CombatType = 0x2C98; // [SEED]; Offsets.h::Hero::CombatType; via AIBaseClientHero::CombatType
    constexpr auto FollowerTargetDelay = 0x2DB8; // [SEED]; Offsets.h::Hero::FollowerTargetDelay; via AIBaseClientHero::FollowerTargetDelay
    constexpr auto Exp = 0x4CF0; // [SEED]; Offsets.h::Hero::Exp; via AIBaseClientHero::Exp
    constexpr auto LevelRef = 0x4D18; // [SEED]; Offsets.h::Hero::LevelRef; via AIBaseClientHero::LevelRef
    constexpr auto LevelUpPoints = 0x4D78; // [SEED]; Offsets.h::Hero::LevelUpPoints; via Seed.Hero::LevelUpPoints
    constexpr auto VisionScore = 0x55E0; // [SEED]; Offsets.h::Hero::VisionScore; via AIBaseClientHero::VisionScore
    constexpr auto ShutdownValue = 0x5608; // [SEED]; Offsets.h::Hero::ShutdownValue; via AIBaseClientHero::ShutdownValue
    constexpr auto BaseGoldOnDeath = 0x5630; // [SEED]; Offsets.h::Hero::BaseGoldOnDeath; via AIBaseClientHero::BaseGoldOnDeath
    constexpr auto NeutralMinionsKilled = 0x5658; // [SEED]; Offsets.h::Hero::NeutralMinionsKilled; via AIBaseClientHero::NeutralMinionsKilled
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto BuffManagerOffset = 0x28B8; // [SEED]; Offsets.h::BuffManager::Offset; via AIBaseClientBuffManager::Offset
    constexpr auto BuffEntriesEnd = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntriesEnd
    constexpr auto BuffEntryBuff = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntryBuff
    constexpr auto BuffType = 0xC; // [S]; stable layout; via AIBaseClientBuffManager::BuffType
    constexpr auto BuffNamePtr = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::BuffNamePtr
    constexpr auto BuffNameStr = 0x8; // [S]; stable layout; via AIBaseClientBuffManager::BuffNameStr
    constexpr auto BuffStartTime = 0x18; // [S]; stable layout; via AIBaseClientBuffManager::BuffStartTime
    constexpr auto BuffEndTime = 0x1C; // [S]; stable layout; via AIBaseClientBuffManager::BuffEndTime
    constexpr auto BuffStacksAlt = 0x38; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacksAlt
    constexpr auto BuffStacks = 0x78; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacks
    constexpr auto AiManagerOffset = 0x41F0; // [SEED]; Offsets.h::AiManager::Offset; via Seed.AiManager::Offset
    constexpr auto AiManagerInnerManager = 0x10; // [SEED]; Offsets.h::AiManager::InnerManager; via Seed.AiManager::InnerManager
    constexpr auto TargetPosition = 0x34; // [SEED]; Offsets.h::AiManager::TargetPosition; via Seed.AiManager::TargetPosition
    constexpr auto Velocity = 0x318; // [SEED]; Offsets.h::AiManager::Velocity; via Seed.AiManager::Velocity
    constexpr auto IsMoving = 0x31C; // [SEED]; Offsets.h::AiManager::IsMoving; via Seed.AiManager::IsMoving
    constexpr auto CurrentSegment = 0x320; // [SEED]; Offsets.h::AiManager::CurrentSegment; via Seed.AiManager::CurrentSegment
    constexpr auto PathStart = 0x330; // [SEED]; Offsets.h::AiManager::PathStart; via Seed.AiManager::PathStart
    constexpr auto PathEndFallback = 0x33C; // [SEED]; Offsets.h::AiManager::PathEnd; via Seed.AiManager::PathEnd
    constexpr auto Segments = 0x348; // [SEED]; Offsets.h::AiManager::Segments; via Seed.AiManager::Segments
    constexpr auto SegmentsCount = 0x350; // [SEED]; Offsets.h::AiManager::SegmentsCount; via Seed.AiManager::SegmentsCount
    constexpr auto DashSpeed = 0x360; // [SEED]; Offsets.h::AiManager::DashSpeed; via Seed.AiManager::DashSpeed
    constexpr auto IsDashing = 0x384; // [SEED]; Offsets.h::AiManager::IsDashing; via Seed.AiManager::IsDashing
    constexpr auto ServerPos = 0x474; // [SEED]; Offsets.h::AiManager::ServerPos; via Seed.AiManager::ServerPos
    constexpr auto MoveVec3 = 0x480; // [SEED]; Offsets.h::AiManager::MoveVec3; via Seed.AiManager::MoveVec3
} // namespace AIHeroClient

namespace AIMinionClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto MP = 0x360; // [SEED]; Offsets.h::Mana::MP; via AIBaseClientMana::MP
    constexpr auto MaxMP = 0x388; // [SEED]; Offsets.h::Mana::MaxMP; via AIBaseClientMana::MaxMP
    constexpr auto PAR = 0xE00; // [SEED]; Offsets.h::Mana::PAR; via AIBaseClientMana::PAR
    constexpr auto MaxPAR = 0xE28; // [INFERRED]; PAR + 0x28 inferred LeagueObfuscation pair for MaxPAR; via AIBaseClientMana::MaxPAR
    constexpr auto SAR = 0x108; // [SEED]; Offsets.h::Mana::SAR; via AIBaseClientMana::SAR
    constexpr auto MaxSAR = 0x130; // [SEED]; Offsets.h::Mana::MaxSAR; via AIBaseClientMana::MaxSAR
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto HeroStatsBase = 0x1B88; // [SEED]; Offsets.h::HeroStats::Base; via AIBaseClientHeroStats::Base
    constexpr auto PercentCooldownMod = 0x1B88; // [SEED]; Offsets.h::HeroStats::PercentCooldownMod; via AIBaseClientHeroStats::PercentCooldownMod
    constexpr auto AbilityHaste = 0x1BB0; // [SEED]; Offsets.h::HeroStats::AbilityHaste; via AIBaseClientHeroStats::AbilityHaste
    constexpr auto PercentCooldownCapMod = 0x1BD8; // [SEED]; Offsets.h::HeroStats::PercentCooldownCapMod; via AIBaseClientHeroStats::PercentCooldownCapMod
    constexpr auto PassiveCdEndTime = 0x1C00; // [SEED]; Offsets.h::HeroStats::PassiveCdEndTime; via AIBaseClientHeroStats::PassiveCdEndTime
    constexpr auto PassiveCdTotalTime = 0x1C28; // [SEED]; Offsets.h::HeroStats::PassiveCdTotalTime; via AIBaseClientHeroStats::PassiveCdTotalTime
    constexpr auto FlatPhysicalDmgMod = 0x1CC8; // [SEED]; Offsets.h::HeroStats::FlatPhysicalDmgMod; via AIBaseClientHeroStats::FlatPhysicalDmgMod
    constexpr auto PercentPhysicalDmgMod = 0x1CF0; // [SEED]; Offsets.h::HeroStats::PercentPhysicalDmgMod; via AIBaseClientHeroStats::PercentPhysicalDmgMod
    constexpr auto PercentBonusPhysDmgMod = 0x1D18; // [SEED]; Offsets.h::HeroStats::PercentBonusPhysDmgMod; via AIBaseClientHeroStats::PercentBonusPhysDmgMod
    constexpr auto PercentBasePhysDmgFlat = 0x1D40; // [SEED]; Offsets.h::HeroStats::PercentBasePhysDmgFlat; via AIBaseClientHeroStats::PercentBasePhysDmgFlat
    constexpr auto FlatMagicDmgMod = 0x1D68; // [SEED]; Offsets.h::HeroStats::FlatMagicDmgMod; via AIBaseClientHeroStats::FlatMagicDmgMod
    constexpr auto PercentMagicDmgMod = 0x1D90; // [SEED]; Offsets.h::HeroStats::PercentMagicDmgMod; via AIBaseClientHeroStats::PercentMagicDmgMod
    constexpr auto FlatMagicReduction = 0x1DB8; // [SEED]; Offsets.h::HeroStats::FlatMagicReduction; via AIBaseClientHeroStats::FlatMagicReduction
    constexpr auto PercentMagicReduction = 0x1DE0; // [SEED]; Offsets.h::HeroStats::PercentMagicReduction; via AIBaseClientHeroStats::PercentMagicReduction
    constexpr auto FlatCastRangeMod = 0x1E08; // [SEED]; Offsets.h::HeroStats::FlatCastRangeMod; via AIBaseClientHeroStats::FlatCastRangeMod
    constexpr auto AttackSpeedMod = 0x1E30; // [SEED]; Offsets.h::HeroStats::AttackSpeedMod; via AIBaseClientHeroStats::AttackSpeedMod
    constexpr auto PercentAttackSpeedMod = 0x1E58; // [SEED]; Offsets.h::HeroStats::PercentAttackSpeedMod; via AIBaseClientHeroStats::PercentAttackSpeedMod
    constexpr auto PercentMultiAtkSpeedMod = 0x1E80; // [SEED]; Offsets.h::HeroStats::PercentMultiAtkSpeedMod; via AIBaseClientHeroStats::PercentMultiAtkSpeedMod
    constexpr auto PercentHealingAmountMod = 0x1EA8; // [SEED]; Offsets.h::HeroStats::PercentHealingAmountMod; via AIBaseClientHeroStats::PercentHealingAmountMod
    constexpr auto BaseAttackDamage = 0x1ED0; // [SEED]; Offsets.h::HeroStats::BaseAttackDamage; via AIBaseClientHeroStats::BaseAttackDamage
    constexpr auto BaseAtkDmgSansScale = 0x1EF8; // [SEED]; Offsets.h::HeroStats::BaseAtkDmgSansScale; via AIBaseClientHeroStats::BaseAtkDmgSansScale
    constexpr auto FlatBaseAtkDmgMod = 0x1F20; // [SEED]; Offsets.h::HeroStats::FlatBaseAtkDmgMod; via AIBaseClientHeroStats::FlatBaseAtkDmgMod
    constexpr auto PercentBaseAtkDmgMod = 0x1F48; // [SEED]; Offsets.h::HeroStats::PercentBaseAtkDmgMod; via AIBaseClientHeroStats::PercentBaseAtkDmgMod
    constexpr auto BaseAbilityDamage = 0x1F70; // [SEED]; Offsets.h::HeroStats::BaseAbilityDamage; via AIBaseClientHeroStats::BaseAbilityDamage
    constexpr auto CritDamageMultiplier = 0x1F98; // [SEED]; Offsets.h::HeroStats::CritDamageMultiplier; via AIBaseClientHeroStats::CritDamageMultiplier
    constexpr auto ScaleSkinCoef = 0x1FC0; // [SEED]; Offsets.h::HeroStats::ScaleSkinCoef; via AIBaseClientHeroStats::ScaleSkinCoef
    constexpr auto Dodge = 0x1FE8; // [SEED]; Offsets.h::HeroStats::Dodge; via AIBaseClientHeroStats::Dodge
    constexpr auto Crit = 0x2010; // [SEED]; Offsets.h::HeroStats::Crit; via AIBaseClientHeroStats::Crit
    constexpr auto Armor = 0x2060; // [SEED]; Offsets.h::HeroStats::Armor; via AIBaseClientHeroStats::Armor
    constexpr auto BonusArmor = 0x2088; // [SEED]; Offsets.h::HeroStats::BonusArmor; via AIBaseClientHeroStats::BonusArmor
    constexpr auto SpellBlock = 0x20B0; // [SEED]; Offsets.h::HeroStats::SpellBlock; via AIBaseClientHeroStats::SpellBlock
    constexpr auto BonusSpellBlock = 0x20D8; // [SEED]; Offsets.h::HeroStats::BonusSpellBlock; via AIBaseClientHeroStats::BonusSpellBlock
    constexpr auto HPRegenRate = 0x2100; // [SEED]; Offsets.h::HeroStats::HPRegenRate; via AIBaseClientHeroStats::HPRegenRate
    constexpr auto BaseHPRegenRate = 0x2128; // [SEED]; Offsets.h::HeroStats::BaseHPRegenRate; via AIBaseClientHeroStats::BaseHPRegenRate
    constexpr auto MoveSpeed = 0x2150; // [SEED]; Offsets.h::HeroStats::MoveSpeed; via AIBaseClientHeroStats::MoveSpeed
    constexpr auto MoveSpeedBaseIncrease = 0x2178; // [SEED]; Offsets.h::HeroStats::MoveSpeedBaseIncrease; via AIBaseClientHeroStats::MoveSpeedBaseIncrease
    constexpr auto AttackRange = 0x21A0; // [SEED]; Offsets.h::HeroStats::AttackRange; via AIBaseClientHeroStats::AttackRange
    constexpr auto FlatBubbleRadiusMod = 0x21C8; // [SEED]; Offsets.h::HeroStats::FlatBubbleRadiusMod; via AIBaseClientHeroStats::FlatBubbleRadiusMod
    constexpr auto PercentBubbleRadiusMod = 0x21F0; // [SEED]; Offsets.h::HeroStats::PercentBubbleRadiusMod; via AIBaseClientHeroStats::PercentBubbleRadiusMod
    constexpr auto FlatArmorPen = 0x2218; // [SEED]; Offsets.h::HeroStats::FlatArmorPen; via AIBaseClientHeroStats::FlatArmorPen
    constexpr auto PhysicalLethality = 0x2240; // [SEED]; Offsets.h::HeroStats::PhysicalLethality; via AIBaseClientHeroStats::PhysicalLethality
    constexpr auto PercentArmorPen = 0x2268; // [SEED]; Offsets.h::HeroStats::PercentArmorPen; via AIBaseClientHeroStats::PercentArmorPen
    constexpr auto PercentBonusArmorPen = 0x2290; // [SEED]; Offsets.h::HeroStats::PercentBonusArmorPen; via AIBaseClientHeroStats::PercentBonusArmorPen
    constexpr auto PercentCritBonusArmorPen = 0x22B8; // [SEED]; Offsets.h::HeroStats::PercentCritBonusArmorPen; via AIBaseClientHeroStats::PercentCritBonusArmorPen
    constexpr auto PercentCritTotalArmorPen = 0x22E0; // [SEED]; Offsets.h::HeroStats::PercentCritTotalArmorPen; via AIBaseClientHeroStats::PercentCritTotalArmorPen
    constexpr auto FlatMagicPen = 0x2308; // [SEED]; Offsets.h::HeroStats::FlatMagicPen; via AIBaseClientHeroStats::FlatMagicPen
    constexpr auto MagicLethality = 0x2330; // [SEED]; Offsets.h::HeroStats::MagicLethality; via AIBaseClientHeroStats::MagicLethality
    constexpr auto PercentMagicPen = 0x2358; // [SEED]; Offsets.h::HeroStats::PercentMagicPen; via AIBaseClientHeroStats::PercentMagicPen
    constexpr auto PercentBonusMagicPen = 0x2380; // [SEED]; Offsets.h::HeroStats::PercentBonusMagicPen; via AIBaseClientHeroStats::PercentBonusMagicPen
    constexpr auto PercentLifeSteal = 0x23A8; // [SEED]; Offsets.h::HeroStats::PercentLifeSteal; via AIBaseClientHeroStats::PercentLifeSteal
    constexpr auto PercentSpellVamp = 0x23D0; // [SEED]; Offsets.h::HeroStats::PercentSpellVamp; via AIBaseClientHeroStats::PercentSpellVamp
    constexpr auto PercentOmnivamp = 0x23F8; // [SEED]; Offsets.h::HeroStats::PercentOmnivamp; via AIBaseClientHeroStats::PercentOmnivamp
    constexpr auto PercentPhysicalVamp = 0x2420; // [SEED]; Offsets.h::HeroStats::PercentPhysicalVamp; via AIBaseClientHeroStats::PercentPhysicalVamp
    constexpr auto PathfindingRadiusMod = 0x2448; // [SEED]; Offsets.h::HeroStats::PathfindingRadiusMod; via AIBaseClientHeroStats::PathfindingRadiusMod
    constexpr auto PercentCCReduction = 0x2470; // [SEED]; Offsets.h::HeroStats::PercentCCReduction; via AIBaseClientHeroStats::PercentCCReduction
    constexpr auto PercentEXPBonus = 0x2498; // [SEED]; Offsets.h::HeroStats::PercentEXPBonus; via AIBaseClientHeroStats::PercentEXPBonus
    constexpr auto FlatBaseArmorMod = 0x24C0; // [SEED]; Offsets.h::HeroStats::FlatBaseArmorMod; via AIBaseClientHeroStats::FlatBaseArmorMod
    constexpr auto FlatBaseSpellBlockMod = 0x24E8; // [SEED]; Offsets.h::HeroStats::FlatBaseSpellBlockMod; via AIBaseClientHeroStats::FlatBaseSpellBlockMod
    constexpr auto PARRegenRate = 0x2510; // [SEED]; Offsets.h::HeroStats::PARRegenRate; via AIBaseClientHeroStats::PARRegenRate
    constexpr auto PrimaryARBaseRegenRate = 0x2538; // [SEED]; Offsets.h::HeroStats::PrimaryARBaseRegenRate; via AIBaseClientHeroStats::PrimaryARBaseRegenRate
    constexpr auto SecondaryARRegenRate = 0x2560; // [SEED]; Offsets.h::HeroStats::SecondaryARRegenRate; via AIBaseClientHeroStats::SecondaryARRegenRate
    constexpr auto SecondaryARBaseRegenRate = 0x2588; // [SEED]; Offsets.h::HeroStats::SecondaryARBaseRegenRate; via AIBaseClientHeroStats::SecondaryARBaseRegenRate
    constexpr auto FlatBaseAttackSpeedMod = 0x25B0; // [SEED]; Offsets.h::HeroStats::FlatBaseAttackSpeedMod; via AIBaseClientHeroStats::FlatBaseAttackSpeedMod
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
    constexpr auto SpellBookOffset = 0x30E8; // [SEED]; Offsets.h::SpellBook::Offset; via AIBaseClientSpellBook::Offset
    constexpr auto SpellSlotArray = 0xAE0; // [SEED]; Offsets.h::SpellBook::SpellSlotArray; via AIBaseClientSpellBook::SpellSlotArray
    constexpr auto ActiveSpellCast = 0x3120; // [SEED]; Offsets.h::SpellBook::ActiveSpellCast; via AIBaseClientSpellBook::ActiveSpellCast
    constexpr auto SlotLevel = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SlotLevel
    constexpr auto SlotCooldown = 0x30; // [S]; stable layout; via AIBaseClientSpellBook::SlotCooldown
    constexpr auto SlotStacks = 0x5C; // [S]; stable layout; via AIBaseClientSpellBook::SlotStacks
    constexpr auto SlotTotalCd = 0x74; // [S]; stable layout; via AIBaseClientSpellBook::SlotTotalCd
    constexpr auto SlotSpellInput = 0x120; // [SEED]; Offsets.h::SpellBook::SlotSpellInput; via AIBaseClientSpellBook::SlotSpellInput
    constexpr auto SlotSpellInfo = 0x128; // [SEED]; Offsets.h::SpellBook::SlotSpellInfo; via AIBaseClientSpellBook::SlotSpellInfo
    constexpr auto InputTargetNetId = 0x14; // [S]; stable layout; via AIBaseClientSpellBook::InputTargetNetId
    constexpr auto InputStartPos = 0x18; // [S]; stable layout; via AIBaseClientSpellBook::InputStartPos
    constexpr auto InputEndPos = 0x24; // [S]; stable layout; via AIBaseClientSpellBook::InputEndPos
    constexpr auto InfoSpellData = 0x60; // [S]; stable layout; via AIBaseClientSpellBook::InfoSpellData
    constexpr auto DataSpellName = 0x80; // [S]; stable layout; via AIBaseClientSpellBook::DataSpellName
    constexpr auto SpellInfoNamePtr = 0x28; // [S]; stable layout; via AIBaseClientSpellBook::SpellInfoNamePtr
    constexpr auto DataManaCost = 0x5F4; // [S]; stable layout; via AIBaseClientSpellBook::DataManaCost
    constexpr auto DataResource = 0x8; // [S]; stable layout; via AIBaseClientSpellBook::DataResource
    constexpr auto BuffManagerOffset = 0x28B8; // [SEED]; Offsets.h::BuffManager::Offset; via AIBaseClientBuffManager::Offset
    constexpr auto BuffEntriesEnd = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntriesEnd
    constexpr auto BuffEntryBuff = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::EntryBuff
    constexpr auto BuffType = 0xC; // [S]; stable layout; via AIBaseClientBuffManager::BuffType
    constexpr auto BuffNamePtr = 0x10; // [S]; stable layout; via AIBaseClientBuffManager::BuffNamePtr
    constexpr auto BuffNameStr = 0x8; // [S]; stable layout; via AIBaseClientBuffManager::BuffNameStr
    constexpr auto BuffStartTime = 0x18; // [S]; stable layout; via AIBaseClientBuffManager::BuffStartTime
    constexpr auto BuffEndTime = 0x1C; // [S]; stable layout; via AIBaseClientBuffManager::BuffEndTime
    constexpr auto BuffStacksAlt = 0x38; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacksAlt
    constexpr auto BuffStacks = 0x78; // [S]; stable layout; via AIBaseClientBuffManager::BuffStacks
    constexpr auto AiManagerOffset = 0x41F0; // [SEED]; Offsets.h::AiManager::Offset; via Seed.AiManager::Offset
    constexpr auto AiManagerInnerManager = 0x10; // [SEED]; Offsets.h::AiManager::InnerManager; via Seed.AiManager::InnerManager
    constexpr auto TargetPosition = 0x34; // [SEED]; Offsets.h::AiManager::TargetPosition; via Seed.AiManager::TargetPosition
    constexpr auto Velocity = 0x318; // [SEED]; Offsets.h::AiManager::Velocity; via Seed.AiManager::Velocity
    constexpr auto IsMoving = 0x31C; // [SEED]; Offsets.h::AiManager::IsMoving; via Seed.AiManager::IsMoving
    constexpr auto CurrentSegment = 0x320; // [SEED]; Offsets.h::AiManager::CurrentSegment; via Seed.AiManager::CurrentSegment
    constexpr auto PathStart = 0x330; // [SEED]; Offsets.h::AiManager::PathStart; via Seed.AiManager::PathStart
    constexpr auto PathEndFallback = 0x33C; // [SEED]; Offsets.h::AiManager::PathEnd; via Seed.AiManager::PathEnd
    constexpr auto Segments = 0x348; // [SEED]; Offsets.h::AiManager::Segments; via Seed.AiManager::Segments
    constexpr auto SegmentsCount = 0x350; // [SEED]; Offsets.h::AiManager::SegmentsCount; via Seed.AiManager::SegmentsCount
    constexpr auto DashSpeed = 0x360; // [SEED]; Offsets.h::AiManager::DashSpeed; via Seed.AiManager::DashSpeed
    constexpr auto IsDashing = 0x384; // [SEED]; Offsets.h::AiManager::IsDashing; via Seed.AiManager::IsDashing
    constexpr auto ServerPos = 0x474; // [SEED]; Offsets.h::AiManager::ServerPos; via Seed.AiManager::ServerPos
    constexpr auto MoveVec3 = 0x480; // [SEED]; Offsets.h::AiManager::MoveVec3; via Seed.AiManager::MoveVec3
    constexpr auto LaneArray = 0x68; // [SEED]; Offsets.h::Minion::LaneArray; via Seed.Minion::LaneArray
    constexpr auto LaneCount = 0x70; // [SEED]; Offsets.h::Minion::LaneCount; via Seed.Minion::LaneCount
    constexpr auto LaneType = 0x4CC9; // [SEED]; Offsets.h::Minion::LaneType; via Seed.Minion::LaneType
    constexpr auto JungleTypeOffset = 0x4A84; // [SEED]; Offsets.h::JungleType::TypeOffset; via Seed.JungleType::TypeOffset
    constexpr auto TypeFlagsField = 0x4C; // [SEED]; Offsets.h::TypeFlags::ObfuscatedField; via Seed.TypeFlags::ObfuscatedField
} // namespace AIMinionClient

namespace MissileClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto SpellDataPtr = 0x128; // [SEED]; Offsets.h::Missile::SpellDataPtr; via Seed.Missile::SpellDataPtr
    constexpr auto CastInfoBase = 0x2C0; // [SEED]; Offsets.h::Missile::CastInfoBase; via Seed.Missile::CastInfoBase
    constexpr auto SpellName = 0x2E0; // [SEED]; Offsets.h::Missile::SpellName; via Seed.Missile::SpellName
    constexpr auto MissileName = 0x308; // [SEED]; Offsets.h::Missile::MissileName; via Seed.Missile::MissileName
    constexpr auto CasterNetId = 0x358; // [SEED]; Offsets.h::Missile::CasterNetId; via Seed.Missile::CasterNetId
    constexpr auto TargetNetId = 0x35C; // [SEED]; Offsets.h::Missile::TargetNetId; via Seed.Missile::TargetNetId
    constexpr auto MissileNetId = 0x364; // [SEED]; Offsets.h::Missile::MissileNetId; via Seed.Missile::MissileNetId
    constexpr auto StartPos = 0x388; // [SEED]; Offsets.h::Missile::StartPos; via Seed.Missile::StartPos
    constexpr auto EndPos = 0x394; // [SEED]; Offsets.h::Missile::EndPos; via Seed.Missile::EndPos
    constexpr auto CastEndPos = 0x3A4; // [SEED]; Offsets.h::Missile::CastEndPos; via Seed.Missile::CastEndPos
    constexpr auto SpellCastInfoSpellData = 0x0; // [SEED]; Offsets.h::SpellCastInfo::SpellData; via Raw.SpellCastInfo::SpellData
    constexpr auto SpellCastInfoSrcIndex = 0x98; // [S]; stable layout; via Raw.SpellCastInfo::SrcIndex
    constexpr auto SpellCastInfoStartPos = 0xD8; // [S]; stable layout; via Raw.SpellCastInfo::StartPos
    constexpr auto SpellCastInfoEndPos = 0xE4; // [S]; stable layout; via Raw.SpellCastInfo::EndPos
    constexpr auto SpellCastInfoCastPos = 0xF0; // [S]; stable layout; via Raw.SpellCastInfo::CastPos
    constexpr auto SpellCastInfoTargetIndex = 0x108; // [S]; stable layout; via Raw.SpellCastInfo::TargetIndex
    constexpr auto SpellCastInfoDestIndex = 0x108; // [ALIAS]; alias of TargetIndex; via Raw.SpellCastInfo::DestIndex
    constexpr auto SpellCastInfoCastDelay = 0x118; // [S]; stable layout; via Raw.SpellCastInfo::CastDelay
    constexpr auto SpellCastInfoIsAuto = 0x141; // [S]; stable layout; via Raw.SpellCastInfo::IsAuto
    constexpr auto SpellCastInfoSlot = 0x14C; // [S]; stable layout; via Raw.SpellCastInfo::Slot
} // namespace MissileClient

namespace Static {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto EffectEmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via AIBaseClientDirect::EffectEmitter
    constexpr auto MissileClientHandle = 0x2D8; // [SEED]; Offsets.h::GameObject::MissileClient; via AIBaseClientDirect::MissileClient
    constexpr auto ItemList = 0x4D20; // [SEED]; Offsets.h::GameObject::ItemList; via AIBaseClientDirect::ItemList
} // namespace Static

namespace AITurretClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto PhysDmgPercent = 0xE78; // [SEED]; Offsets.h::DamageModifier::PhysDmgPercent; via AIBaseClientDamageModifier::PhysDmgPercent
    constexpr auto MagicDmgPercent = 0xEA0; // [SEED]; Offsets.h::DamageModifier::MagicDmgPercent; via AIBaseClientDamageModifier::MagicDmgPercent
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace AITurretClient

namespace EffectEmitter {
    constexpr auto EmitterHandle = 0x258; // [SEED]; Offsets.h::GameObject::EffectEmitter; via Seed.GameObject::EffectEmitter
    constexpr auto Data = 0x8; // [C]; chimera_structures.h::EffectEmitter::Data; via Supp.EffectEmitter::Data
    constexpr auto Attachment = 0x38; // [C]; chimera_structures.h::EffectEmitter::Attachment; via Supp.EffectEmitter::Attachment
    constexpr auto TargetAttachment = 0x48; // [C]; chimera_structures.h::EffectEmitter::TargetAttachment; via Supp.EffectEmitter::TargetAttachment
    constexpr auto AttachmentData = 0x8; // [C]; chimera_structures.h::EffectEmitterAttachment::Data; via Supp.EffectEmitterAttachment::Data
    constexpr auto AttachmentObject = 0x0; // [C]; chimera_structures.h::EffectEmitterAttachment::Object; via Supp.EffectEmitterAttachment::Object
    constexpr auto OrientationRight = 0x118; // [C]; chimera_structures.h::EffectEmitterData::OrientationRight; via Supp.EffectEmitterData::OrientationRight
    constexpr auto OrientationUp = 0x128; // [C]; chimera_structures.h::EffectEmitterData::OrientationUp; via Supp.EffectEmitterData::OrientationUp
    constexpr auto OrientationForward = 0x138; // [C]; chimera_structures.h::EffectEmitterData::OrientationForward; via Supp.EffectEmitterData::OrientationForward
} // namespace EffectEmitter

namespace BarracksDampenerClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace BarracksDampenerClient

namespace HQClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
    constexpr auto HP = 0x1080; // [SEED]; Offsets.h::Health::HP; via AIBaseClientHealth::HP
    constexpr auto MaxHP = 0x10A8; // [SEED]; Offsets.h::Health::MaxHP; via AIBaseClientHealth::MaxHP
    constexpr auto HPMaxPenalty = 0x10D0; // [SEED]; Offsets.h::Health::HPMaxPenalty; via AIBaseClientHealth::HPMaxPenalty
    constexpr auto AllShield = 0x1120; // [SEED]; Offsets.h::Health::AllShield; via AIBaseClientHealth::AllShield
    constexpr auto PhysicalShield = 0x1148; // [SEED]; Offsets.h::Health::PhysicalShield; via AIBaseClientHealth::PhysicalShield
    constexpr auto MagicalShield = 0x1170; // [SEED]; Offsets.h::Health::MagicalShield; via AIBaseClientHealth::MagicalShield
    constexpr auto ChampSpecific = 0x1198; // [SEED]; Offsets.h::Health::ChampSpecific; via AIBaseClientHealth::ChampSpecific
    constexpr auto InHealAllied = 0x11C0; // [SEED]; Offsets.h::Health::InHealAllied; via AIBaseClientHealth::InHealAllied
    constexpr auto InHealEnemy = 0x11E8; // [SEED]; Offsets.h::Health::InHealEnemy; via AIBaseClientHealth::InHealEnemy
    constexpr auto InDamage = 0x1210; // [SEED]; Offsets.h::Health::InDamage; via AIBaseClientHealth::InDamage
    constexpr auto StopShieldFade = 0x1238; // [SEED]; Offsets.h::Health::StopShieldFade; via AIBaseClientHealth::StopShieldFade
    constexpr auto IsTargetable = 0xED0; // [SEED]; Offsets.h::Targetable::IsTargetable; via AIBaseClientTargetable::IsTargetable
    constexpr auto TargetableFlags = 0xEF8; // [SEED]; Offsets.h::Targetable::TargetableFlags; via AIBaseClientTargetable::TargetableFlags
    constexpr auto ActionState1 = 0x1470; // [SEED]; Offsets.h::ActionState::State1; via Seed.ActionState::State1
    constexpr auto ActionState2 = 0x14A8; // [SEED]; Offsets.h::ActionState::State2; via Seed.ActionState::State2
    constexpr auto Lifetime = 0xDB0; // [SEED]; Offsets.h::Lifetime::Lifetime; via AIBaseClientLifetime::Lifetime
    constexpr auto MaxLifetime = 0xDD8; // [SEED]; Offsets.h::Lifetime::MaxLifetime; via AIBaseClientLifetime::MaxLifetime
    constexpr auto LifetimeTicks = 0xE00; // [SEED]; Offsets.h::Lifetime::LifetimeTicks; via AIBaseClientLifetime::LifetimeTicks
} // namespace HQClient

namespace ShopClient {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
} // namespace ShopClient

namespace Obj_SpawnPoint {
    constexpr auto Index = 0x10; // [SEED]; Offsets.h::GameObject::Index; via AIBaseClientDirect::Index
    constexpr auto Team = 0x3C; // [SEED]; Offsets.h::GameObject::Team; via AIBaseClientDirect::Team
    constexpr auto Name = 0x58; // [SEED]; Offsets.h::GameObject::Name; via AIBaseClientDirect::Name
    constexpr auto NetId = 0xCC; // [SEED]; Offsets.h::GameObject::NetId; via AIBaseClientDirect::NetId
    constexpr auto Dead = 0x250; // [SEED]; Offsets.h::GameObject::Dead; via AIBaseClientDirect::Dead
    constexpr auto Position = 0x25C; // [SEED]; Offsets.h::GameObject::Position; via AIBaseClientDirect::Position
    constexpr auto Visibility = 0x2E0; // [SEED]; Offsets.h::GameObject::Visibility; via AIBaseClientDirect::Visibility
    constexpr auto Visible = 0x308; // [SEED]; Offsets.h::GameObject::Visible; via AIBaseClientDirect::Visible
    constexpr auto Radius = 0x6F8; // [SEED]; Offsets.h::GameObject::Radius; via AIBaseClientDirect::Radius
    constexpr auto CharacterData = 0x40C8; // [SEED]; Offsets.h::GameObject::CharacterData; via AIBaseClientDirect::CharacterData
    constexpr auto CharacterName = 0x4330; // [SEED]; Offsets.h::GameObject::CharacterName; via AIBaseClientDirect::CharacterName
    constexpr auto Direction = 0x21D8; // [SEED]; Offsets.h::GameObject::Direction; via AIBaseClientDirect::Direction
} // namespace Obj_SpawnPoint
}
Where can I get that file, man?
znob is offline

Old 24th March 2026, 01:20 PM   #12995
caitou2024
n00bie

caitou2024's Avatar

Join Date: Apr 2024
Posts: 3
Reputation: 10
Rep Power: 49
caitou2024 has made posts that are generally average in quality
Points: 1,428, Level: 3
Points: 1,428, Level: 3 Points: 1,428, Level: 3 Points: 1,428, Level: 3
Level up: 4%, 672 Points needed
Level up: 4% Level up: 4% Level up: 4%
Activity: 4.8%
Activity: 4.8% Activity: 4.8% Activity: 4.8%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
use signature plugin for get it
hello, this function 'GetSpellCastInfo'
The parameter rcx of this function is the character object, and the rdx is '0x40'. I want to obtain SpellCastInfo for skills such as Q, how should I obtain it
caitou2024 is offline

Old 26th March 2026, 11:22 AM   #12996
andrey1818
n00bie

andrey1818's Avatar

Join Date: Feb 2026
Posts: 2
Reputation: 10
Rep Power: 3
andrey1818 has made posts that are generally average in quality
Points: 73, Level: 1
Points: 73, Level: 1 Points: 73, Level: 1 Points: 73, Level: 1
Level up: 19%, 327 Points needed
Level up: 19% Level up: 19% Level up: 19%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
How can they access the memory of the locked kernel?
Quote:
Originally Posted by caitou2024 View Post
hello, this function 'GetSpellCastInfo'
The parameter rcx of this function is the character object, and the rdx is '0x40'. I want to obtain SpellCastInfo for skills such as Q, how should I obtain it
How can they access the memory of the locked kernel?
andrey1818 is offline

Old 26th March 2026, 03:38 PM   #12997
dacuigege
Junior Member

dacuigege's Avatar

Join Date: Jan 2023
Posts: 42
Reputation: 211
Rep Power: 78
dacuigege has just realized Source Code isnt a magazinedacuigege has just realized Source Code isnt a magazinedacuigege has just realized Source Code isnt a magazine
Points: 2,641, Level: 4
Points: 2,641, Level: 4 Points: 2,641, Level: 4 Points: 2,641, Level: 4
Level up: 78%, 159 Points needed
Level up: 78% Level up: 78% Level up: 78%
Activity: 2.4%
Activity: 2.4% Activity: 2.4% Activity: 2.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Could you please tell me if there are any updated addresses among these? I want to check the latest one as it caused my app to crash.

# define oObjIndex 0x18+0x8 //
# define oSpellIRange1 0x484 //
# define oMissileSpellInfo 0x2B8+0x8 //
# define oMissileSrcIdx 0x350+0x8 //
# define oMissileDestIdx 0x3C0+0x8 //
# define oMissileStartPos 0x380+0x8 //
# define oMissileEndPos 0x38C+0x8 //
dacuigege is offline

Old 29th March 2026, 12:00 AM   #12998
a768787747
n00bie

a768787747's Avatar

Join Date: Sep 2023
Posts: 13
Reputation: 10
Rep Power: 64
a768787747 has made posts that are generally average in quality
Points: 2,004, Level: 3
Points: 2,004, Level: 3 Points: 2,004, Level: 3 Points: 2,004, Level: 3
Level up: 87%, 96 Points needed
Level up: 87% Level up: 87% Level up: 87%
Activity: 2.8%
Activity: 2.8% Activity: 2.8% Activity: 2.8%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by trankhanhtinh1 View Post
my dump not check
Code:
// ===================== GLOBAL POINTERS (RVA) =====================
# define oLocalPlayer            0x1DE1EC0
# define oHeroList               0x1DA8C78
# define oHerroList              0x1DA8CD0
# define oGametime               0x1DB6E70
# define oMissileList            0x1DACA70
# define oNavGrid                0x1DAC9E0   // 48 8B 05 ? ? ? ? 0F 28 DA
# define IssueOrderFlag          0x1D0CF28
# define CastSpellFlag           0x1D0CEC0   // C6 05 ? ? ? ? ? E8 ? ? ? ? 8B 50 ?

// ===================== GAME OBJECT STRUCT =====================
# define oObjName                0x68        // char* / std::string - object name
# define oObjNetId               0xCC        // int32
# define oDead                   0x250       // byte (1=dead, 0=alive)
# define oObjPosition            0x25C       // vec3 (X, Y+4, Z+8)
# define TeamID                  0x259       // byte - team ID
# define oVisibility             0x300       // byte
# define oMana                   0x358       // float (LeagueObfuscation<float>)
# define oMaxMana                0x380       // float (LeagueObfuscation<float>)
# define oObjRadius              0x6F8       // float
# define oTargetable             0xED0       // byte (1=targetable)
// --- Health Stats (LeagueObfuscation<float>, 40-byte entries) ---
// Found via mHP/mMaxHP replication in sub_2E8F10
# define oHealth                 0x1080      // float - current HP (mHP)
# define oMaxHealth              0x10A8      // float - max HP (mMaxHP)
# define oHPMaxPenalty           0x10D0      // float - HP max penalty
# define oAllShield              0x1120      // float - total shield
# define oPhysicalShield         0x1148      // float - physical shield
# define oMagicalShield          0x1170      // float - magical shield
# define NamePlayer              0x4328      // std::string (SSO)

// ===================== BASIC ATTACK =====================
// Chain: obj + oBasicAttackBase -> ptr + oBasicAttackOffset1 -> ptr2 + oBasicAttackOffset2 -> attack data
# define oBasicAttackBase        0x2C68      // 48 8B 81 ? ? ? ? C3 CC 40 56 48 (old: 0x2C90)
# define oBasicAttackOffset1     0x2C0       // 5B C3 CC 48 8B 81 ? ? ? ? C3
# define oBasicAttackOffset2     0x70        // 48 8B 41 ? 48 89 02 4C 89 42 08 F0 41 (old: 0x38)
# define oObjBasicAttackCastCount 0x4CD8     // int32 - incremented per attack (old: 0x4D44)

// ===================== MINION =====================
# define LaneMinionArray         0x68        // 48 8B 46 ? 8B 4E ? ? ? ? ? 48 3B C2
# define LaneMinionCount         0x70        // count following array ptr
# define LaneMinionType          0x4C71      // 0F B6 81 ? ? ? ? 3C ? 74 ? 2C ? 3C ? 76 (old: 0x4CC9)
// LaneMinionType values: 4=Melee, 5=Ranged, 6=Cannon, 7=Super

// ===================== SPELLBOOK (from GameObject) =====================
# define oObjSpellBook           0x30E8
# define oObjOnCastingSpell      0x3120      // SpellBook + 0x38

// ===================== SPELLBOOK INTERNAL =====================
# define oObjSpellBookSpellSlot  0xAE0       // Q=0, W=1, E=2, R=3, D=4, F=5

// ===================== SPELLSLOT STRUCT =====================
# define oSpellSlotLevel         0x28        // int32
# define oSpellSlotSpellInfo     0x128       // ptr to SpellInfo

// ===================== SPELLINFO / SPELLDATA CHAIN =====================
# define oSpellInfoSpellData     0x60        // SpellInfo -> SpellData
# define oSpellDataNameHash      0x24        // int32 FNV hash
# define oSpellDataResource      0x60        // SpellData -> SpellDataResource

// ===================== BUFF MANAGER =====================
// BuffManager = obj + oObjBuffManager
// Iterate: for (ptr = *(BuffMgr+0x18); ptr < *(BuffMgr+0x20); ptr += entrySize)
# define oObjBuffManager         0x2E40      // 48 8D 8B ? ? ? ? 48 8B D7 E8 (old: 0x2E68)
# define oBuffManagerArray       0x18        // BuffManager -> array start ptr
# define oBuffManagerArrayEnd    0x20        // BuffManager -> array end ptr
// BuffInstance struct:
# define oBuffInstanceType       0x08        // byte - buff type (24,25,26 = valid)
# define oBuffInstanceScript     0x10        // ptr -> BuffScript
# define oBuffInstanceStartTime  0x18        // float - start game time
# define oBuffInstanceEndTime    0x1C        // float - end game time (25000+ = permanent)
# define oBuffInstanceStackCount 0x38        // int - instance count (1 = active)
# define oBuffInstanceCount      0x8C        // int - actual stack count
// BuffScript struct:
# define oBuffScriptName         0x08        // char* - buff name string

// ===================== HUD & INPUT =====================
# define oHudInstance            0x1DA8E18   // 48 8B 0D ? ? ? ? 48 85 c9 74 ? 48 8b 49 ? 48 8d
# define oHudInstanceCamera      0x18
# define oHudInstanceInput       0x28
# define oHudMouseVec3           0x34        // F3 0F 10 4F ? 48 8B BC 24 (world mouse pos)
# define oHudInstanceUserData    0x60
# define oHudInstanceSpellInfo   0x68        // 48 8B 48 ? 48 85 C9 74 ? 48 8B 51 ? 48 85 D2 75

// ===================== CHAT SYSTEM =====================
// ChatUI component (accessed from HUD/GUI system):
//   chatMode = *(byte*)(ChatUI + oChatOpenFlag)
//   0 = chat closed (normal game input)
//   1 = team chat open
//   2 = all chat open
//   if (chatMode != 0) → block orbwalker keys
# define oChatOpenFlag           0x288       // byte - chat mode state in ChatUI component

// ===================== FUNCTION ADDRESSES (RVA) =====================
# define oIssueOrder             0x2A6460
# define oCastSpellWrapper       0x5C8CE0    // 48 89 48 ? 55 56 57 41 54 41 55 (old: 0x949DE0)
# define AttackDelay             0x567E10    // E8 ? ? ? ? 33 C0 F3 0F 11 83 (old: 0x540DB0)
# define oGetAttackWindup        0x567D10    // 48 89 5C 24 ? 48 89 74 24 ? 57 48 83 EC 60... (old: 0x540CB0)
# define oGetBoundingRadius      0x28BBF0
# define oGetCollisionFlags      0x11C94D0
# define oGetAiManager           0x28A030
# define GetPing                 0x692C20    // E8 ? ? ? ? 8B F8 39 03 (old: 0x667DB0)
# define WorldToScreen           0x12759A0   // E8 ? ? ? ? F3 0F 10 44 24 ? F3 41 0F 11 06 (old: 0x13CF720)
# define isMinion                0x30E250    // E8 ? ? ? ? 48 8B 0B F3 0F 10 41 (old: 0x301120)
# define isTurret                0x30E0F0    // 40 53 48 83 EC 20 48 8B D9 48 85 C9 74 27 (old: 0x300FC0)
# define GetFirstObject          0x532EF0    // 48 83 EC ? 48 8B 51 ? 8B 41 ? 48 8D 0C C2 (old: 0x513E40)
# define GetNextObject           0x533CD0    // 0F B7 42 ? 44 8B 41 (old: 0x5034B0)

// ===================== MISSILE OBJECT =====================
# define oMissileSpellCastPtr    0x8         // ptr to external SpellCastInfo
# define oMissileCastInfo        0x2C0       // embedded CastInfo base
# define oMissileSpellDataInst   0x2C0       // first QWORD = SpellDataInst ptr
# define oMissileSpellName       0x2E0       // std::string
# define oMissileMissileName     0x308       // std::string
# define oMissileStartPos        0x330       // vec3
# define oMissileEndPos          0x33C       // vec3
# define oMissileCastEndPos      0x34C       // vec3
# define oMissileCasterNetId     0x358       // int32
# define oMissileNetworkId       0x364       // int32
# define oMissilePosition        0x25C       // vec3 (current, inherited)

// ===================== AI MANAGER =====================
// AiManager*=*(obj + oObjAiManager)
// All offsets below are relative to AiManager pointer
// Fields are XOR-encrypted (use LeagueObfuscation Decrypt<T>)
# define oObjAiManager               0x4028  // ptr (old: 0x41A8)

// --- AiManager Internal ---
# define oAiManagerNavPathPtr        0x30    // ptr to NavigationPath (returned by GetAiManager)
# define oAiManagerRefCount          0x1F0   // int32

// --- Movement Data (relative to AiManager) ---
// All patterns VERIFIED in binary via byte scan
# define oAiManagerVelocity          0x318   // float (movement speed scalar)
# define oAiManagerIsMoving          0x31C   // float/bool
# define oAiManagerCurrentSegment    0x320   // int32

// --- Path Data ---
# define oAiManagerStartPath         0x330   // vec3 (X, Y+4, Z+8) - path start position
# define oAiManagerEndPath           0x33C   // vec3 (X, Y+4, Z+8) - path end position
# define oAiManagerTargetPosition    0x33C   // = oAiManagerStartPath + 0xC

// --- Navigation ---
# define oAiManagerNavArray          0x348   // ptr to waypoints vec3 array
# define oAiManagerSegmentsCount     0x350   // int32

// --- Dash ---
# define oAiManagerDashSpeed         0x360   // float
# define oAiManagerIsDashing         0x384   // byte/bool

// --- Server Position ---
# define oAiManagerServerPos         0x474   // vec3 (X, Y+4, Z+8)

// --- Derived ---
# define oAiManagerMoveVec3          0x480   // vec3 (oAiManagerStartPath + 0x150)
# define oAiManagerHasPath           0x350   // SegmentsCount > 0

// ===================== NAV GRID =====================
// NavGrid*= *(oNavGrid)
// NavGridManager* =*(NavGrid + 0x8)
// Used for: IsWall check, IsBush check, pathfinding collision
# define oNavGridManager             0x8     // ptr to NavGridManager from NavGrid global
# define oNavGridWidth               0x708   // int32 - grid width (cells)
# define oNavGridHeight              0x70C   // int32 - grid height (cells)
# define oNavGridScale               0x714   // float - cell scale factor
# define oNavGridMinX                0xEC    // float - world min X coordinate
# define oNavGridMinZ                0xF4    // float - world min Z coordinate
# define oNavGridData                0x150   // ptr to byte[] - cell flags array
// IsWall/IsBush check:
//   cellX = (int)((posX - minX) *scale)
//   cellZ = (int)((posZ - minZ)* scale)
//   index = cellZ *width + cellX
//   flags =*(gridData + index)
//   isBush = (flags != 0)
//   isWall = check collision flags via oGetCollisionFlags
Set the wall at the designated position, I want to achieve automatic avoidance of the wall by right-clicking, please help.

Does anyone know how to set a coordinate as a function of the wall，Thanks

Does anyone know how to set a coordinate as a function of the wall，Thanks

Does anyone know how to set a coordinate as a function of the wall，Thanks
a768787747 is offline

Old 30th March 2026, 03:10 PM   #12999
HitMeHarder
n00bie

HitMeHarder's Avatar

Join Date: Sep 2018
Posts: 1
Reputation: 10
Rep Power: 185
HitMeHarder has made posts that are generally average in quality
Points: 1, Level: 1
Points: 1, Level: 1 Points: 1, Level: 1 Points: 1, Level: 1
Level up: 0%, 1 Points needed
Level up: 0% Level up: 0% Level up: 0%
Activity: 0%
Activity: 0% Activity: 0% Activity: 0%
Does anyone have the latest offsets for mac osx league? Or know how to get it? I tried the static dumper on another thread but it's not working well and not providing good results, I also have no idea what the offsets or string names are for the mac osx offsets so I don't know what I'm even looking for.
HitMeHarder is offline

Old 1st April 2026, 09:51 AM   #13000
oPillow
n00bie

oPillow's Avatar

Join Date: Oct 2021
Posts: 13
Reputation: 10
Rep Power: 110
oPillow has made posts that are generally average in quality
Points: 3,313, Level: 5
Points: 3,313, Level: 5 Points: 3,313, Level: 5 Points: 3,313, Level: 5
Level up: 65%, 287 Points needed
Level up: 65% Level up: 65% Level up: 65%
Activity: 2.8%
Activity: 2.8% Activity: 2.8% Activity: 2.8%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Hey guys! I'm working on an ability tracker in my external, beside detecting when spells are casted I'm having no problems except getting correct cooldowns for ultimate spell and summoner spells, I have no clue how ultimate haste and summoner haste is stored, I thought maybe if I have like 50 haste then my ultimate haste would be 70 in memory (for example, depending on items and champs) but I scanned the player memory many times to find anything that should be ultimate haste/summoner haste and didn't find anything, so now I suspect it's probably some modifier like your ultimate haste is 1.2 * haste so the 1.2 would be in memory. But before scanning for it I wanted to ask if anyone knows anything about this to help me before potentially wasting my time on it! (I know I could just check my items and runes and calculate it manually but I want a more solid way)

Here is my partial code if anyone is interested:

Code:
bool CachedActor::isSpellReady(const ESpellIndex spellIndex) const
{
 return getSpellTime(spellIndex) <= 0.0f;
}

float CachedActor::getSpellTime(const ESpellIndex spellIndex) const
{
 const float elapsed = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(spellCooldowns.at(static_cast<int>(spellIndex)).getTime()).count()) / 1000.0f;
 const float cooldown = getSpellCooldown(spellIndex);

 if (cooldown <= 0.0f)
 {
  return 0.0f;
 }
 
 return std::max(0.0f, cooldown - elapsed);
}

float CachedActor::getSpellCooldown(const ESpellIndex spellIndex) const
{
 const auto spells = getSpellSlots();

 if (static_cast<int>(spellIndex) < spells.size())
 {
  const auto& spell = spells.at(static_cast<int>(spellIndex));
 
  if (spellIndex == ESpellIndex::D || spellIndex == ESpellIndex::F)
  {
   const auto spellDetails = Client::instance->getModuleManager()->getGame()->getAlternativeSpellDetails(String("Default"), spell.getSpellInfo().getName());
 
   if (!spellDetails.name.empty() && spellDetails.cooldownTime.size() == 1)
   {
    return spellDetails.cooldownTime.at(0);
   }
  }
  else
  {
   const auto spellDetails = Client::instance->getModuleManager()->getGame()->getSpellDetails(getModelName(), spell.getSpellInfo().getName());
   const auto spellLevel = spell.getSpellLevel();
 
   if (!spellDetails.name.empty() && spellLevel > 0 && spellLevel - 1 < spellDetails.cooldownTime.size())
   {
    if (spellIndex == ESpellIndex::R)
    {
     return spellDetails.cooldownTime.at(spellLevel) / (1.0f + getAbilityHaste() / 100.0f);
    }
    else
    {
     return spellDetails.cooldownTime.at(spellLevel) / (1.0f + getAbilityHaste() / 100.0f);
    }
   }
  }
 }
 
 return 0.0f;
}
Last edited by oPillow; 1st April 2026 at 01:25 PM.
oPillow is offline

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 650 of 651 « First < 150 550 600 640 646 647 648 649 650 651 > 

AD
Tags
typedef, #define, offsets, pobj;, int, float, updated, bool, thread, dword

« Previous Thread | Next Thread »

Forum Jump

    League of Legends

All times are GMT. The time now is 01:45 AM.
Copyright ©2000-2026, Unknowncheats™
DMCA - Contact
Terms of Use - Privacy Policy - Forum Rules

UnknownCheats - Leading the game hacking and cheat development scene since 2000 
UnKnoWnCheaTs Game Hacking Portal UnKnoWnCheaTs Game Hacking Forum – Cheats, Hacks, and Tutorials Download Game Hacks, Cheats and Hacking Tools – UnKnoWnCheaTs Game Hacking Wiki – Tutorials and Guides on UnKnoWnCheaTs Toggle Dark Mode Register at UnKnoWnCheaTs – Join the Greatest Game Hacking Community

Go Back   UnKnoWnCheaTs - Multiplayer Game Hacking and Cheats
MMO and Strategy Games
League of Legends
Reload this Page [Coding] League of Legends Reversal, Structs and Offsets
User Name:
Password:
Remember Me? 

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 651 of 651 « First < 151 551 601 641 647 648 649 650 651 

Thread Tools
Old 2nd April 2026, 04:55 AM   #13001
AaAasen
n00bie

AaAasen's Avatar

Join Date: Mar 2025
Posts: 1
Reputation: 10
Rep Power: 27
AaAasen has made posts that are generally average in quality
Points: 1, Level: 1
Points: 1, Level: 1 Points: 1, Level: 1 Points: 1, Level: 1
Level up: 0%, 1 Points needed
Level up: 0% Level up: 0% Level up: 0%
Activity: 0%
Activity: 0% Activity: 0% Activity: 0%
thank you bro,great job!
AaAasen is offline

Old 3rd April 2026, 06:36 AM   #13002
kyudev
1337 H4x0!2

kyudev's Avatar

Join Date: Dec 2020
Posts: 142
Reputation: 674
Rep Power: 131
kyudev 666kyudev 666kyudev 666kyudev 666kyudev 666kyudev 666
Recognitions
Members who have contributed financial support towards UnKnoWnCheaTs. Donator (1)
Points: 4,912, Level: 7
Points: 4,912, Level: 7 Points: 4,912, Level: 7 Points: 4,912, Level: 7
Level up: 46%, 488 Points needed
Level up: 46% Level up: 46% Level up: 46%
Activity: 9.4%
Activity: 9.4% Activity: 9.4% Activity: 9.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by Alexis913 View Post
Hello everyone,

I'm back after a few months without playing.
I've updated the offsets (many patterns were still valid), but I see that I can't read the ViewMatrix values (0x1E2C030) because it returns an error. I don't have any issues with the other offsets.

Has there been any change to this?

Code:
inline constexpr std::intptr_t pMatrixBase                      = 0x1E2C030;
inline constexpr std::intptr_t oViewMatrix                      = 0x1AC;
inline constexpr std::intptr_t oProjectionMatrix                = 0x22C;

Did you figure this out? I am running into the same issue
__________________
I don’t bypass, I charm the system.
kyudev is offline

Old 4th April 2026, 10:42 PM   #13003
bditt
Supreme G0d

bditt's Avatar

Join Date: Nov 2015
Location: 0xC1A551F1ED
Posts: 394
Reputation: 5456
Rep Power: 261
bditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATS
Recognitions
Members who have contributed financial support towards UnKnoWnCheaTs. Donator (2)
Points: 16,845, Level: 17
Points: 16,845, Level: 17 Points: 16,845, Level: 17 Points: 16,845, Level: 17
Level up: 32%, 955 Points needed
Level up: 32% Level up: 32% Level up: 32%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Can anyone post an updated offset list so I can check to make sure I'm finding the right things please?

EDIT: I'm dumb, just realized there is a publicly released dumper.

EDIT 2: Seems the publicly released dumper isn't fully correct? Can someone posted a checked version for EUW or NA please?
Last edited by bditt; 5th April 2026 at 06:01 AM.
bditt is online now

Old 5th April 2026, 10:59 PM   #13004
thunartx
n00bie

thunartx's Avatar

Join Date: Sep 2024
Posts: 5
Reputation: 10
Rep Power: 39
thunartx has made posts that are generally average in quality
Points: 1,176, Level: 2
Points: 1,176, Level: 2 Points: 1,176, Level: 2 Points: 1,176, Level: 2
Level up: 56%, 224 Points needed
Level up: 56% Level up: 56% Level up: 56%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and Offsets
DrawCircle Offset pls or sig
thunartx is offline

Old 6th April 2026, 09:11 PM   #13005
ibrahimcelik
Junior Member

ibrahimcelik's Avatar

Join Date: Jul 2023
Posts: 51
Reputation: 10
Rep Power: 67
ibrahimcelik has made posts that are generally average in quality
Points: 2,241, Level: 4
Points: 2,241, Level: 4 Points: 2,241, Level: 4 Points: 2,241, Level: 4
Level up: 21%, 559 Points needed
Level up: 21% Level up: 21% Level up: 21%
Activity: 99.0%
Activity: 99.0% Activity: 99.0% Activity: 99.0%
Last Achievements
League of Legends Reversal, Structs and Offsets
// ==================== GLOBAL POINTERS ====================
# define oLocalPlayer 0x1df65a8
# define oHerroList 0x1dbef80
# define oGametime 0x1dcd1e0
// FAILED: oMissileList
# define oNavGrid 0x1dc2d10
# define oHudInstance 0x1dbf0c8
# define oUnderMouseObj 0x1dc2fa0
# define ViewPort 0x1dd29e0
# define IssueOrderFlag 0x1d21d28
# define CastSpellFlag 0x1d21cc0
// FAILED: oMinionManager
# define oObjectManager 0x1dbef28
# define oViewPort2 0x1e8abc8
# define oMySpellState 0x1dc5af8
// FAILED: oKeyBoardHit
# define oMouseScreenVec2 0x1dc2d48

// ==================== FUNCTIONS ====================
# define oIssueOrder 0x2aae10
# define AttackDelay 0x53cfd0
# define GetPing 0x673720
# define WorldToScreen 0x127edc0
// FAILED: isMinion
# define oGetAttackWindup 0x53ced0
# define oGetBoundingRadius 0x290250
# define isTurret 0x315560
# define GetFirstObject 0x526370
# define GetNextObject 0x526db0
# define oCastSpellWrapper 0x1f1280
# define oGetCollisionFlags 0x11d0b20
# define oGetAiManager 0x51ed40
# define oPrintChat 0x860c00
# define IsAlive 0x2f2390
# define IsHero 0x315660
// FAILED: CastSpell2

// ==================== STRUCT OFFSETS (Pattern-based) ====================
// --- AiManager ---
# define oObjAiManager 0x4030
# define oAiManagerStartPath 0x88
# define oAiManagerEndPath 0x88
// --- BasicAttack ---
# define oBasicAttackBase 0x2c68
# define oBasicAttackOffset1 0x2c0
# define oBasicAttackOffset2 0x70
// --- BuffManager ---
# define oObjBuffManager 0x28b8
// --- GameObject ---
# define oObjNetId 0xcc
# define oObjPosition 0x25c
# define oObjRadius 0x6f8
# define TeamID 0x259
# define oTargetable 0xed0
# define NamePlayer 0x4328
// --- HUD ---
# define oHudSpell 0x68
# define oHudMouse 0x28
// --- Health_Verified ---
# define oHealth_base 0x1080
# define oHealth_mHP 0x1080
# define oMaxHealth_mMaxHP 0x10a8
# define oHPMaxPenalty 0x10d0
# define oAllShield 0x1120
# define oPhysicalShield 0x1148
# define oMagicalShield 0x1170
# define oChampSpecificHealth 0x1198
// --- Minion ---
# define LaneMinionArray 0x68
# define LaneMinionType 0x4c71
// --- Missile ---
# define oMissileCastInfo 0x1c0
// --- NavGrid ---
# define oNavGridMinX 0x30
// --- SpellBook ---
# define oObjSpellBook 0x30e8
# define oObjSpellBookSpellSlot 0xae0
// --- SpellData ---
# define oSpellDataResource 0x8
// --- SpellSlot ---
# define oSpellSlotSpellInfo 0x130
// --- StatBlock_Verified ---
# define oHeroStatBase 0x1b88
# define oArmor 0x2060
# define oSpellBlock 0x20b0
# define oBaseAttackDamage 0x1ed0
# define oAttackRange 0x21a0
# define oMoveSpeed 0x2150
# define oAttackSpeedMod 0x1e30
# define oCrit 0x2010
# define oHPRegenRate 0x2100
# define oPercentCooldownMod 0x1b88
# define oAbilityHasteMod 0x1bb0
# define oFlatMagicPenetration 0x2308
# define oPercentMagicPenetration 0x2358
# define oFlatArmorPenetration 0x2218
# define oPercentArmorPenetration 0x2268
# define oPercentLifeSteal 0x23a8
# define oBonusArmor 0x2088
# define oBonusSpellBlock 0x20d8
# define oFlatPhysicalDamageMod 0x1cc8
# define oFlatMagicDamageMod 0x1d68
# define oBaseAbilityDamage 0x1f70

// ================================================================
// REPLICATED PROPERTIES (146 total)
// v3: Offsets are ABSOLUTE from game object (base+delta resolved)
// ================================================================

// --- sub_2043E0 (11 properties) [stat base=0x1b88] ---
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_2045B0 (12 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mPrimaryARRegenRateRep 0x2510 // base 0x1b88 + 0x988

// --- sub_2047A0 (18 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mMoveSpeedBaseIncrease 0x2178 // base 0x1b88 + 0x5f0
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668

// --- sub_204AB0 (27 properties) [stat base=0x1b88] ---
# define o_mPassiveCooldownEndTime 0x1c00 // base 0x1b88 + 0x78
# define o_mPassiveCooldownTotalTime 0x1c28 // base 0x1b88 + 0xa0
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mFlatCastRangeMod 0x1e08 // base 0x1b88 + 0x280
# define o_mPercentCooldownMod 0x1e08 // base 0x1b88 + 0x280
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mBaseAbilityDamage 0x1f70 // base 0x1b88 + 0x3e8
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatArmorPenetration 0x2218 // base 0x1b88 + 0x690
# define o_mPercentArmorPenetration 0x2268 // base 0x1b88 + 0x6e0
# define o_mFlatMagicPenetration 0x2308 // base 0x1b88 + 0x780
# define o_mPercentMagicPenetration 0x2358 // base 0x1b88 + 0x7d0
# define o_mPercentLifeStealMod 0x23a8 // base 0x1b88 + 0x820
# define o_mPercentSpellVampMod 0x23d0 // base 0x1b88 + 0x848
# define o_mPercentPhysicalVamp 0x2420 // base 0x1b88 + 0x898
# define o_mPARRegenRate 0x2510 // base 0x1b88 + 0x988

// --- sub_204EB0 (30 properties) [stat base=0x1b88] ---
# define o_mAbilityHasteMod 0x1bb0 // base 0x1b88 + 0x28
# define o_mPercentCooldownCapMod 0x1bd8 // base 0x1b88 + 0x50
# define o_mPercentBonusPhysicalDamageMod 0x1d18 // base 0x1b88 + 0x190
# define o_mPercentBasePhysicalDamageAsFlatBonusMod 0x1d40 // base 0x1b88 + 0x1b8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mPercentHealingAmountMod 0x1ea8 // base 0x1b88 + 0x320
# define o_mBaseAttackDamageSansPercentScale 0x1ef8 // base 0x1b88 + 0x370
# define o_mFlatBaseAttackDamageMod 0x1f20 // base 0x1b88 + 0x398
# define o_mPercentBaseAttackDamageMod 0x1f48 // base 0x1b88 + 0x3c0
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mFlatBaseHPPoolMod 0x2038 // base 0x1b88 + 0x4b0
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mBaseHPRegenRate 0x2128 // base 0x1b88 + 0x5a0
# define o_mPhysicalLethality 0x2240 // base 0x1b88 + 0x6b8
# define o_mPercentBonusArmorPenetration 0x2290 // base 0x1b88 + 0x708
# define o_mPercentCritBonusArmorPenetration 0x22b8 // base 0x1b88 + 0x730
# define o_mPercentCritTotalArmorPenetration 0x22e0 // base 0x1b88 + 0x758
# define o_mMagicLethality 0x2330 // base 0x1b88 + 0x7a8
# define o_mPercentBonusMagicPenetration 0x2380 // base 0x1b88 + 0x7f8
# define o_mPercentOmnivampMod 0x23f8 // base 0x1b88 + 0x870
# define o_mPercentCCReduction 0x2470 // base 0x1b88 + 0x8e8
# define o_mPercentEXPBonus 0x2498 // base 0x1b88 + 0x910
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mPrimaryARBaseRegenRateRep 0x2538 // base 0x1b88 + 0x9b0
# define o_mSecondaryARRegenRateRep 0x2560 // base 0x1b88 + 0x9d8
# define o_mSecondaryARBaseRegenRateRep 0x2588 // base 0x1b88 + 0xa00
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_205360 (5 properties) [stat base=0x1b88] ---
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668
# define o_mPathfindingRadiusMod 0x2448 // base 0x1b88 + 0x8c0

// --- sub_2EF260 (10 properties) ---
# define o_mMaxHP 0x2800
# define o_mHPMaxPenalty 0x5000
# define o_mAllShield 0xa000
# define o_mPhysicalShield 0xc800
# define o_mMagicalShield 0xf000
# define o_mChampSpecificHealth 0x1180
# define o_mIncomingHealingAllied 0x1400
# define o_mIncomingHealingEnemy 0x1680
# define o_mIncomingDamage 0x1900
# define o_mHP 0x1080

// --- sub_2EF910 (16 properties) ---
# define o_mMaxPAR 0x2800
# define o_mSAR 0x1080
# define o_mMaxSAR 0x1300
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPAR 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mGold 0x2830
# define o_mGoldTotal 0x2858
# define o_mMinimumGold 0x2880
# define o_mExp 0x4ce8
# define o_mVisionScore 0x55d8
# define o_mShutdownValue 0x5600
# define o_mBaseGoldGivenOnDeath 0x5628

// --- sub_2F0E00 (15 properties) ---
// mMaxPAR (offset unknown)
// mPARState (offset unknown)
// mMaxPAR (offset unknown)
// mMaxSAR (offset unknown)
// mMaxSAR (offset unknown)
# define o_mPAR 0x1080
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mPercentDamageToBarracksMinionMod 0x1c50
# define o_mFlatDamageReductionFromBarracksMinionMod 0x1c78
# define o_mIncreasedMoveSpeedMinionMod 0x1ca0
# define o_mFollowerTargetDelay 0x2db8

// --- sub_3717E0 (2 properties) ---
# define o_mMP 0x3600
# define o_mMaxMP 0x3880

// ================================================================
// UNIQUE PROPERTIES SORTED BY OFFSET (97 unique)
// ================================================================
// ???? mPARState (offset unknown)
// 0x2800 mMaxHP
// 0x2800 mMaxPAR
// 0x5000 mHPMaxPenalty
// 0xa000 mAllShield
// 0xc800 mPhysicalShield
// 0xf000 mMagicalShield
// 0x1080 mPAR
// 0x1080 mSAR
// 0x1180 mChampSpecificHealth
// 0x1300 mMaxSAR
// 0x1400 mIncomingHealingAllied
// 0x1680 mIncomingHealingEnemy
// 0x1900 mIncomingDamage
// 0x3600 mMP
// 0x3880 mMaxMP
// 0xdb00 mLifetime
// 0xdd80 mMaxLifetime
// 0xe000 mLifetimeTicks
// 0xe780 mPhysicalDamagePercentageModifier
// 0xea00 mMagicalDamagePercentageModifier
// 0x1080 mHP
// 0x1bb0 mAbilityHasteMod
// 0x1bd8 mPercentCooldownCapMod
// 0x1c00 mPassiveCooldownEndTime
// 0x1c28 mPassiveCooldownTotalTime
// 0x1c50 mPercentDamageToBarracksMinionMod
// 0x1c78 mFlatDamageReductionFromBarracksMinionMod
// 0x1ca0 mIncreasedMoveSpeedMinionMod
// 0x1cc8 mFlatPhysicalDamageMod
// 0x1cf0 mPercentPhysicalDamageMod
// 0x1d18 mPercentBonusPhysicalDamageMod
// 0x1d40 mPercentBasePhysicalDamageAsFlatBonusMod
// 0x1d68 mFlatMagicDamageMod
// 0x1d90 mPercentMagicDamageMod
// 0x1db8 mFlatMagicReduction
// 0x1de0 mPercentMagicReduction
// 0x1e08 mFlatCastRangeMod
// 0x1e08 mPercentCooldownMod
// 0x1e30 mAttackSpeedMod
// 0x1e58 mPercentAttackSpeedMod
// 0x1e80 mPercentMultiplicativeAttackSpeedMod
// 0x1ea8 mPercentHealingAmountMod
// 0x1ed0 mBaseAttackDamage
// 0x1ef8 mBaseAttackDamageSansPercentScale
// 0x1f20 mFlatBaseAttackDamageMod
// 0x1f48 mPercentBaseAttackDamageMod
// 0x1f70 mBaseAbilityDamage
// 0x1f98 mCritDamageMultiplier
// 0x1fc0 mScaleSkinCoef
// 0x1fe8 mDodge
// 0x2010 mCrit
// 0x2038 mFlatBaseHPPoolMod
// 0x2060 mArmor
// 0x2088 mBonusArmor
// 0x20b0 mSpellBlock
// 0x20d8 mBonusSpellBlock
// 0x2100 mHPRegenRate
// 0x2128 mBaseHPRegenRate
// 0x2150 mMoveSpeed
// 0x2178 mMoveSpeedBaseIncrease
// 0x21a0 mAttackRange
// 0x21c8 mFlatBubbleRadiusMod
// 0x21f0 mPercentBubbleRadiusMod
// 0x2218 mFlatArmorPenetration
// 0x2240 mPhysicalLethality
// 0x2268 mPercentArmorPenetration
// 0x2290 mPercentBonusArmorPenetration
// 0x22b8 mPercentCritBonusArmorPenetration
// 0x22e0 mPercentCritTotalArmorPenetration
// 0x2308 mFlatMagicPenetration
// 0x2330 mMagicLethality
// 0x2358 mPercentMagicPenetration
// 0x2380 mPercentBonusMagicPenetration
// 0x23a8 mPercentLifeStealMod
// 0x23d0 mPercentSpellVampMod
// 0x23f8 mPercentOmnivampMod
// 0x2420 mPercentPhysicalVamp
// 0x2448 mPathfindingRadiusMod
// 0x2470 mPercentCCReduction
// 0x2498 mPercentEXPBonus
// 0x24c0 mFlatBaseArmorMod
// 0x24e8 mFlatBaseSpellBlockMod
// 0x2510 mPARRegenRate
// 0x2510 mPrimaryARRegenRateRep
// 0x2538 mPrimaryARBaseRegenRateRep
// 0x2560 mSecondaryARRegenRateRep
// 0x2588 mSecondaryARBaseRegenRateRep
// 0x25b0 mFlatBaseAttackSpeedMod
// 0x2830 mGold
// 0x2858 mGoldTotal
// 0x2880 mMinimumGold
// 0x2db8 mFollowerTargetDelay
// 0x4ce8 mExp
// 0x55d8 mVisionScore
// 0x5600 mShutdownValue
// 0x5628 mBaseGoldGivenOnDeath

League of Legends zoom hack offset;
Default float value = 2250

0x1DBF0C8,
{ 0xA0, 0x18, 0x268, 0x324 }
Last edited by Altoid; 7th April 2026 at 12:09 AM. Reason: Removed asking for rep portion.
ibrahimcelik is online now

Old 7th April 2026, 12:20 AM   #13006
kurobakaito1992
n00bie

kurobakaito1992's Avatar

Join Date: Oct 2014
Posts: 21
Reputation: 10
Rep Power: 280
kurobakaito1992 has made posts that are generally average in quality
Points: 8,433, Level: 10
Points: 8,433, Level: 10 Points: 8,433, Level: 10 Points: 8,433, Level: 10
Level up: 76%, 267 Points needed
Level up: 76% Level up: 76% Level up: 76%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
IssueOrder and CastSpell still working and undetect ?
kurobakaito1992 is offline

Old 7th April 2026, 02:03 AM   #13007
bditt
Supreme G0d

bditt's Avatar

Join Date: Nov 2015
Location: 0xC1A551F1ED
Posts: 394
Reputation: 5456
Rep Power: 261
bditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATSbditt DEFINES UNKNOWNCHEATS
Recognitions
Members who have contributed financial support towards UnKnoWnCheaTs. Donator (2)
Points: 16,845, Level: 17
Points: 16,845, Level: 17 Points: 16,845, Level: 17 Points: 16,845, Level: 17
Level up: 32%, 955 Points needed
Level up: 32% Level up: 32% Level up: 32%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
What game version is this for?
Also do you have MinionManager offset?
Quote:
Originally Posted by ibrahimcelik View Post
// ==================== GLOBAL POINTERS ====================
# define oLocalPlayer 0x1df65a8
# define oHerroList 0x1dbef80
# define oGametime 0x1dcd1e0
// FAILED: oMissileList
# define oNavGrid 0x1dc2d10
# define oHudInstance 0x1dbf0c8
# define oUnderMouseObj 0x1dc2fa0
# define ViewPort 0x1dd29e0
# define IssueOrderFlag 0x1d21d28
# define CastSpellFlag 0x1d21cc0
// FAILED: oMinionManager
# define oObjectManager 0x1dbef28
# define oViewPort2 0x1e8abc8
# define oMySpellState 0x1dc5af8
// FAILED: oKeyBoardHit
# define oMouseScreenVec2 0x1dc2d48

// ==================== FUNCTIONS ====================
# define oIssueOrder 0x2aae10
# define AttackDelay 0x53cfd0
# define GetPing 0x673720
# define WorldToScreen 0x127edc0
// FAILED: isMinion
# define oGetAttackWindup 0x53ced0
# define oGetBoundingRadius 0x290250
# define isTurret 0x315560
# define GetFirstObject 0x526370
# define GetNextObject 0x526db0
# define oCastSpellWrapper 0x1f1280
# define oGetCollisionFlags 0x11d0b20
# define oGetAiManager 0x51ed40
# define oPrintChat 0x860c00
# define IsAlive 0x2f2390
# define IsHero 0x315660
// FAILED: CastSpell2

// ==================== STRUCT OFFSETS (Pattern-based) ====================
// --- AiManager ---
# define oObjAiManager 0x4030
# define oAiManagerStartPath 0x88
# define oAiManagerEndPath 0x88
// --- BasicAttack ---
# define oBasicAttackBase 0x2c68
# define oBasicAttackOffset1 0x2c0
# define oBasicAttackOffset2 0x70
// --- BuffManager ---
# define oObjBuffManager 0x28b8
// --- GameObject ---
# define oObjNetId 0xcc
# define oObjPosition 0x25c
# define oObjRadius 0x6f8
# define TeamID 0x259
# define oTargetable 0xed0
# define NamePlayer 0x4328
// --- HUD ---
# define oHudSpell 0x68
# define oHudMouse 0x28
// --- Health_Verified ---
# define oHealth_base 0x1080
# define oHealth_mHP 0x1080
# define oMaxHealth_mMaxHP 0x10a8
# define oHPMaxPenalty 0x10d0
# define oAllShield 0x1120
# define oPhysicalShield 0x1148
# define oMagicalShield 0x1170
# define oChampSpecificHealth 0x1198
// --- Minion ---
# define LaneMinionArray 0x68
# define LaneMinionType 0x4c71
// --- Missile ---
# define oMissileCastInfo 0x1c0
// --- NavGrid ---
# define oNavGridMinX 0x30
// --- SpellBook ---
# define oObjSpellBook 0x30e8
# define oObjSpellBookSpellSlot 0xae0
// --- SpellData ---
# define oSpellDataResource 0x8
// --- SpellSlot ---
# define oSpellSlotSpellInfo 0x130
// --- StatBlock_Verified ---
# define oHeroStatBase 0x1b88
# define oArmor 0x2060
# define oSpellBlock 0x20b0
# define oBaseAttackDamage 0x1ed0
# define oAttackRange 0x21a0
# define oMoveSpeed 0x2150
# define oAttackSpeedMod 0x1e30
# define oCrit 0x2010
# define oHPRegenRate 0x2100
# define oPercentCooldownMod 0x1b88
# define oAbilityHasteMod 0x1bb0
# define oFlatMagicPenetration 0x2308
# define oPercentMagicPenetration 0x2358
# define oFlatArmorPenetration 0x2218
# define oPercentArmorPenetration 0x2268
# define oPercentLifeSteal 0x23a8
# define oBonusArmor 0x2088
# define oBonusSpellBlock 0x20d8
# define oFlatPhysicalDamageMod 0x1cc8
# define oFlatMagicDamageMod 0x1d68
# define oBaseAbilityDamage 0x1f70

// ================================================================
// REPLICATED PROPERTIES (146 total)
// v3: Offsets are ABSOLUTE from game object (base+delta resolved)
// ================================================================

// --- sub_2043E0 (11 properties) [stat base=0x1b88] ---
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_2045B0 (12 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mPrimaryARRegenRateRep 0x2510 // base 0x1b88 + 0x988

// --- sub_2047A0 (18 properties) [stat base=0x1b88] ---
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mMoveSpeedBaseIncrease 0x2178 // base 0x1b88 + 0x5f0
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668

// --- sub_204AB0 (27 properties) [stat base=0x1b88] ---
# define o_mPassiveCooldownEndTime 0x1c00 // base 0x1b88 + 0x78
# define o_mPassiveCooldownTotalTime 0x1c28 // base 0x1b88 + 0xa0
# define o_mFlatPhysicalDamageMod 0x1cc8 // base 0x1b88 + 0x140
# define o_mPercentPhysicalDamageMod 0x1cf0 // base 0x1b88 + 0x168
# define o_mFlatMagicDamageMod 0x1d68 // base 0x1b88 + 0x1e0
# define o_mPercentMagicDamageMod 0x1d90 // base 0x1b88 + 0x208
# define o_mFlatMagicReduction 0x1db8 // base 0x1b88 + 0x230
# define o_mPercentMagicReduction 0x1de0 // base 0x1b88 + 0x258
# define o_mFlatCastRangeMod 0x1e08 // base 0x1b88 + 0x280
# define o_mPercentCooldownMod 0x1e08 // base 0x1b88 + 0x280
# define o_mAttackSpeedMod 0x1e30 // base 0x1b88 + 0x2a8
# define o_mBaseAttackDamage 0x1ed0 // base 0x1b88 + 0x348
# define o_mBaseAbilityDamage 0x1f70 // base 0x1b88 + 0x3e8
# define o_mDodge 0x1fe8 // base 0x1b88 + 0x460
# define o_mCrit 0x2010 // base 0x1b88 + 0x488
# define o_mArmor 0x2060 // base 0x1b88 + 0x4d8
# define o_mSpellBlock 0x20b0 // base 0x1b88 + 0x528
# define o_mHPRegenRate 0x2100 // base 0x1b88 + 0x578
# define o_mAttackRange 0x21a0 // base 0x1b88 + 0x618
# define o_mFlatArmorPenetration 0x2218 // base 0x1b88 + 0x690
# define o_mPercentArmorPenetration 0x2268 // base 0x1b88 + 0x6e0
# define o_mFlatMagicPenetration 0x2308 // base 0x1b88 + 0x780
# define o_mPercentMagicPenetration 0x2358 // base 0x1b88 + 0x7d0
# define o_mPercentLifeStealMod 0x23a8 // base 0x1b88 + 0x820
# define o_mPercentSpellVampMod 0x23d0 // base 0x1b88 + 0x848
# define o_mPercentPhysicalVamp 0x2420 // base 0x1b88 + 0x898
# define o_mPARRegenRate 0x2510 // base 0x1b88 + 0x988

// --- sub_204EB0 (30 properties) [stat base=0x1b88] ---
# define o_mAbilityHasteMod 0x1bb0 // base 0x1b88 + 0x28
# define o_mPercentCooldownCapMod 0x1bd8 // base 0x1b88 + 0x50
# define o_mPercentBonusPhysicalDamageMod 0x1d18 // base 0x1b88 + 0x190
# define o_mPercentBasePhysicalDamageAsFlatBonusMod 0x1d40 // base 0x1b88 + 0x1b8
# define o_mPercentAttackSpeedMod 0x1e58 // base 0x1b88 + 0x2d0
# define o_mPercentMultiplicativeAttackSpeedMod 0x1e80 // base 0x1b88 + 0x2f8
# define o_mPercentHealingAmountMod 0x1ea8 // base 0x1b88 + 0x320
# define o_mBaseAttackDamageSansPercentScale 0x1ef8 // base 0x1b88 + 0x370
# define o_mFlatBaseAttackDamageMod 0x1f20 // base 0x1b88 + 0x398
# define o_mPercentBaseAttackDamageMod 0x1f48 // base 0x1b88 + 0x3c0
# define o_mCritDamageMultiplier 0x1f98 // base 0x1b88 + 0x410
# define o_mFlatBaseHPPoolMod 0x2038 // base 0x1b88 + 0x4b0
# define o_mBonusArmor 0x2088 // base 0x1b88 + 0x500
# define o_mBonusSpellBlock 0x20d8 // base 0x1b88 + 0x550
# define o_mBaseHPRegenRate 0x2128 // base 0x1b88 + 0x5a0
# define o_mPhysicalLethality 0x2240 // base 0x1b88 + 0x6b8
# define o_mPercentBonusArmorPenetration 0x2290 // base 0x1b88 + 0x708
# define o_mPercentCritBonusArmorPenetration 0x22b8 // base 0x1b88 + 0x730
# define o_mPercentCritTotalArmorPenetration 0x22e0 // base 0x1b88 + 0x758
# define o_mMagicLethality 0x2330 // base 0x1b88 + 0x7a8
# define o_mPercentBonusMagicPenetration 0x2380 // base 0x1b88 + 0x7f8
# define o_mPercentOmnivampMod 0x23f8 // base 0x1b88 + 0x870
# define o_mPercentCCReduction 0x2470 // base 0x1b88 + 0x8e8
# define o_mPercentEXPBonus 0x2498 // base 0x1b88 + 0x910
# define o_mFlatBaseArmorMod 0x24c0 // base 0x1b88 + 0x938
# define o_mFlatBaseSpellBlockMod 0x24e8 // base 0x1b88 + 0x960
# define o_mPrimaryARBaseRegenRateRep 0x2538 // base 0x1b88 + 0x9b0
# define o_mSecondaryARRegenRateRep 0x2560 // base 0x1b88 + 0x9d8
# define o_mSecondaryARBaseRegenRateRep 0x2588 // base 0x1b88 + 0xa00
# define o_mFlatBaseAttackSpeedMod 0x25b0 // base 0x1b88 + 0xa28

// --- sub_205360 (5 properties) [stat base=0x1b88] ---
# define o_mScaleSkinCoef 0x1fc0 // base 0x1b88 + 0x438
# define o_mMoveSpeed 0x2150 // base 0x1b88 + 0x5c8
# define o_mFlatBubbleRadiusMod 0x21c8 // base 0x1b88 + 0x640
# define o_mPercentBubbleRadiusMod 0x21f0 // base 0x1b88 + 0x668
# define o_mPathfindingRadiusMod 0x2448 // base 0x1b88 + 0x8c0

// --- sub_2EF260 (10 properties) ---
# define o_mMaxHP 0x2800
# define o_mHPMaxPenalty 0x5000
# define o_mAllShield 0xa000
# define o_mPhysicalShield 0xc800
# define o_mMagicalShield 0xf000
# define o_mChampSpecificHealth 0x1180
# define o_mIncomingHealingAllied 0x1400
# define o_mIncomingHealingEnemy 0x1680
# define o_mIncomingDamage 0x1900
# define o_mHP 0x1080

// --- sub_2EF910 (16 properties) ---
# define o_mMaxPAR 0x2800
# define o_mSAR 0x1080
# define o_mMaxSAR 0x1300
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPAR 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mGold 0x2830
# define o_mGoldTotal 0x2858
# define o_mMinimumGold 0x2880
# define o_mExp 0x4ce8
# define o_mVisionScore 0x55d8
# define o_mShutdownValue 0x5600
# define o_mBaseGoldGivenOnDeath 0x5628

// --- sub_2F0E00 (15 properties) ---
// mMaxPAR (offset unknown)
// mPARState (offset unknown)
// mMaxPAR (offset unknown)
// mMaxSAR (offset unknown)
// mMaxSAR (offset unknown)
# define o_mPAR 0x1080
# define o_mLifetime 0xdb00
# define o_mMaxLifetime 0xdd80
# define o_mLifetimeTicks 0xe000
# define o_mPhysicalDamagePercentageModifier 0xe780
# define o_mMagicalDamagePercentageModifier 0xea00
# define o_mPercentDamageToBarracksMinionMod 0x1c50
# define o_mFlatDamageReductionFromBarracksMinionMod 0x1c78
# define o_mIncreasedMoveSpeedMinionMod 0x1ca0
# define o_mFollowerTargetDelay 0x2db8

// --- sub_3717E0 (2 properties) ---
# define o_mMP 0x3600
# define o_mMaxMP 0x3880

// ================================================================
// UNIQUE PROPERTIES SORTED BY OFFSET (97 unique)
// ================================================================
// ???? mPARState (offset unknown)
// 0x2800 mMaxHP
// 0x2800 mMaxPAR
// 0x5000 mHPMaxPenalty
// 0xa000 mAllShield
// 0xc800 mPhysicalShield
// 0xf000 mMagicalShield
// 0x1080 mPAR
// 0x1080 mSAR
// 0x1180 mChampSpecificHealth
// 0x1300 mMaxSAR
// 0x1400 mIncomingHealingAllied
// 0x1680 mIncomingHealingEnemy
// 0x1900 mIncomingDamage
// 0x3600 mMP
// 0x3880 mMaxMP
// 0xdb00 mLifetime
// 0xdd80 mMaxLifetime
// 0xe000 mLifetimeTicks
// 0xe780 mPhysicalDamagePercentageModifier
// 0xea00 mMagicalDamagePercentageModifier
// 0x1080 mHP
// 0x1bb0 mAbilityHasteMod
// 0x1bd8 mPercentCooldownCapMod
// 0x1c00 mPassiveCooldownEndTime
// 0x1c28 mPassiveCooldownTotalTime
// 0x1c50 mPercentDamageToBarracksMinionMod
// 0x1c78 mFlatDamageReductionFromBarracksMinionMod
// 0x1ca0 mIncreasedMoveSpeedMinionMod
// 0x1cc8 mFlatPhysicalDamageMod
// 0x1cf0 mPercentPhysicalDamageMod
// 0x1d18 mPercentBonusPhysicalDamageMod
// 0x1d40 mPercentBasePhysicalDamageAsFlatBonusMod
// 0x1d68 mFlatMagicDamageMod
// 0x1d90 mPercentMagicDamageMod
// 0x1db8 mFlatMagicReduction
// 0x1de0 mPercentMagicReduction
// 0x1e08 mFlatCastRangeMod
// 0x1e08 mPercentCooldownMod
// 0x1e30 mAttackSpeedMod
// 0x1e58 mPercentAttackSpeedMod
// 0x1e80 mPercentMultiplicativeAttackSpeedMod
// 0x1ea8 mPercentHealingAmountMod
// 0x1ed0 mBaseAttackDamage
// 0x1ef8 mBaseAttackDamageSansPercentScale
// 0x1f20 mFlatBaseAttackDamageMod
// 0x1f48 mPercentBaseAttackDamageMod
// 0x1f70 mBaseAbilityDamage
// 0x1f98 mCritDamageMultiplier
// 0x1fc0 mScaleSkinCoef
// 0x1fe8 mDodge
// 0x2010 mCrit
// 0x2038 mFlatBaseHPPoolMod
// 0x2060 mArmor
// 0x2088 mBonusArmor
// 0x20b0 mSpellBlock
// 0x20d8 mBonusSpellBlock
// 0x2100 mHPRegenRate
// 0x2128 mBaseHPRegenRate
// 0x2150 mMoveSpeed
// 0x2178 mMoveSpeedBaseIncrease
// 0x21a0 mAttackRange
// 0x21c8 mFlatBubbleRadiusMod
// 0x21f0 mPercentBubbleRadiusMod
// 0x2218 mFlatArmorPenetration
// 0x2240 mPhysicalLethality
// 0x2268 mPercentArmorPenetration
// 0x2290 mPercentBonusArmorPenetration
// 0x22b8 mPercentCritBonusArmorPenetration
// 0x22e0 mPercentCritTotalArmorPenetration
// 0x2308 mFlatMagicPenetration
// 0x2330 mMagicLethality
// 0x2358 mPercentMagicPenetration
// 0x2380 mPercentBonusMagicPenetration
// 0x23a8 mPercentLifeStealMod
// 0x23d0 mPercentSpellVampMod
// 0x23f8 mPercentOmnivampMod
// 0x2420 mPercentPhysicalVamp
// 0x2448 mPathfindingRadiusMod
// 0x2470 mPercentCCReduction
// 0x2498 mPercentEXPBonus
// 0x24c0 mFlatBaseArmorMod
// 0x24e8 mFlatBaseSpellBlockMod
// 0x2510 mPARRegenRate
// 0x2510 mPrimaryARRegenRateRep
// 0x2538 mPrimaryARBaseRegenRateRep
// 0x2560 mSecondaryARRegenRateRep
// 0x2588 mSecondaryARBaseRegenRateRep
// 0x25b0 mFlatBaseAttackSpeedMod
// 0x2830 mGold
// 0x2858 mGoldTotal
// 0x2880 mMinimumGold
// 0x2db8 mFollowerTargetDelay
// 0x4ce8 mExp
// 0x55d8 mVisionScore
// 0x5600 mShutdownValue
// 0x5628 mBaseGoldGivenOnDeath

League of Legends zoom hack offset;
Default float value = 2250

0x1DBF0C8,
{ 0xA0, 0x18, 0x268, 0x324 }
bditt is online now

Old 7th April 2026, 02:18 PM   #13008
ibrahimcelik
Junior Member

ibrahimcelik's Avatar

Join Date: Jul 2023
Posts: 51
Reputation: 10
Rep Power: 67
ibrahimcelik has made posts that are generally average in quality
Points: 2,241, Level: 4
Points: 2,241, Level: 4 Points: 2,241, Level: 4 Points: 2,241, Level: 4
Level up: 21%, 559 Points needed
Level up: 21% Level up: 21% Level up: 21%
Activity: 99.0%
Activity: 99.0% Activity: 99.0% Activity: 99.0%
Last Achievements
League of Legends Reversal, Structs and Offsets
for latest. Btw, MinionManager = 0x1F2BEF80;
I need drawcircle
ibrahimcelik is online now

Old 7th April 2026, 04:49 PM   #13009
kyudev
1337 H4x0!2

kyudev's Avatar

Join Date: Dec 2020
Posts: 142
Reputation: 674
Rep Power: 131
kyudev 666kyudev 666kyudev 666kyudev 666kyudev 666kyudev 666
Recognitions
Members who have contributed financial support towards UnKnoWnCheaTs. Donator (1)
Points: 4,912, Level: 7
Points: 4,912, Level: 7 Points: 4,912, Level: 7 Points: 4,912, Level: 7
Level up: 46%, 488 Points needed
Level up: 46% Level up: 46% Level up: 46%
Activity: 9.4%
Activity: 9.4% Activity: 9.4% Activity: 9.4%
Last Achievements
League of Legends Reversal, Structs and OffsetsLeague of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by ibrahimcelik View Post
// ==================== GLOBAL POINTERS ====================
# define oLocalPlayer 0x1df65a8
# define oHerroList 0x1dbef80
# define oGametime 0x1dcd1e0
// FAILED: oMissileList
# define oNavGrid 0x1dc2d10
# define oHudInstance 0x1dbf0c8
# define oUnderMouseObj 0x1dc2fa0
# define ViewPort 0x1dd29e0
# define IssueOrderFlag 0x1d21d28
# define CastSpellFlag 0x1d21cc0
...

May be a stupid question but is "isDead" gone?
I couldnt find it with reclass + the dumper returns a wrong offset + last few posts its either not there or wrong :/
__________________
I don’t bypass, I charm the system.
Last edited by kyudev; 7th April 2026 at 04:50 PM. Reason: spelling
kyudev is offline

Old 7th April 2026, 05:02 PM   #13010
sq834960394
h4x0!2

sq834960394's Avatar

Join Date: Dec 2023
Location: Tokyo
Posts: 92
Reputation: 627
Rep Power: 58
sq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiend
Points: 2,765, Level: 4
Points: 2,765, Level: 4 Points: 2,765, Level: 4 Points: 2,765, Level: 4
Level up: 95%, 35 Points needed
Level up: 95% Level up: 95% Level up: 95%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and Offsets
there are some address i'm using now,,,
work hard, guys~
Code:
# pragma once
# include"pch.h"

inline  uint64_t ads_LOL = 0;  //Must be initialized first at dllmain.

# define BaseAddr(x) ((x) + (ads_LOL))
//26DF40

# ifdef __CN

//KeyBoard and Mouse dinput vTableHooks:
# define ads_ppKeyBoardDevice 0x1E6CC18 //16.7  48 8B 0D ? ? ? ? 4C 8D 0D ? ? ? ? 4C 8B 05 ? ? ? ? BA 上一条  FF 50 50 83 F8 01 0F 87
# define ads_pKeyBoard_rgdod  0x1E6CC28 //16.7  r8:4C 8B 05 ? ? ? ? BA ? ? ? ? C7 第一条 
# define ads_pKeyBoard_pdwInOut  0x1E6CC48 //16.7  r9:4C 8B 05 ? ? ? ? BA ? ? ? ? C7 第一条
# define ads_ppMouseDevice 0x1E6CC20  //16.7  48 8B 0D ? ? ? ? 4C 8D 0D ? ? ? ? 4C 8B 05 ? ? ? ? BA 下一条  FF 50 50 83 F8 01 77
# define ads_oVtable 0x0     //下面的mov rax,[rcx]
# define ads_oGetDeviceData 0x50  //不会变，
# define ads_oGetDeviceState 0x48 //这是GetDeviceState的函数pattern：83 BB ? ? ? ? 00 74 ? F6 43 ? 01 75 14 F6(0x48)

//D3D hook:
# define D3dRender  0x1E8E038        // 16.7  48 8B 05 ? ? ? ? 48 8B 98 ? ? ? ? 48 8B CB 48
# define D3DPresento3  0x40      // 15.12 FF 50 ? E8 ? ? ? ? 0F
# define D3DoPresento2  0x0      // 15.112 48 8B 01 FF 50 ? E8 ? ? ? ? 0F 45 F8
# define D3DoPresento1  0x200     // 15.12_ 48 83 B9 ? ? ? ? 00 48 8B D9 75 ? 48 8B 89 第一条
# define D3DoScreenSize 0x120

//GetCursorPos hook:
# define ads_IAT_GetCursorPos 0x18CA9B0 //16.5.1  FF 15 ? ? ? ? 48 8D 54 24 ? 48 8B CB FF 中间那条
// FF 25 1F 00 00 00 ->  //FF 25 00 00 00 00 40 6A 55 32 FD 7F 00 00
# define ads_MouseScreenVec2 0x1DC6D08 //16.7  48 8B 0D ? ? ? ? E8 ? ? ? ? 48 8B 0D ? ? ? ? 48 8B 01 第一条
# define ads_oMouseScreenVec2_x 0xC  //8B 70 ? 44 8B C6 44 8B 70
# define ads_oMouseScreenVec2_y 0x10  //41 8B 41 ? 81 C2

//Ver = 15.16.704.1925, CN, PUBLIC
//#define ModuleBaseAddr ((uint64_t)GetModuleHandle(NULL))
# define ModuleBaseAddr (Global::LOLBaseAddress)
//#define BaseAddr(x) (uint64_t)((x) + (ModuleBaseAddr))
# define Spoof_Trampoline 0x15AC05    // 15.18 FF 23
# define ThreadBegineTrampoline 0x182C9FC  // 15.1 FF E1 C3
# define SendPackageACEhook 0x100017000   // 15.16
# define KeyboardEventACEhook 0x100015000  // 18   1C
namespace Offset
{

 namespace Data
 {
 
 
  //KeyBoard
  constexpr uint64_t KeyBoardHit = 0x1E6B3D0;     // 16.7  C6 84 ? ? ? ? ? 01 C6 84 ? ? ? ? ? 01 89
 
  constexpr float MAP_SIZE = 14830.0f;  // 或 15000.0f，根据实际情况调整
 
  // CRT(first instruction): 48 83 EC 28 FF 15 9E FF FF
  constexpr uint64_t GameTime = 0x1DD11A0;                    // 16.7   F3 0F 5C 35 ? ? ? ? 0F 28 F8   
 
  //MiniMap:
  constexpr uint64_t MiniMapInstance = 0x1DC9A90;    // 16.7 4C 8B 0D ? ? ? ? 48 8B 1D ? ? ? ? 40 32 FF
  constexpr uint64_t oHudMiniMap = 0x4E0;      // 16.7 48 8B ? ? ? ? ? 48 85 C9 74 ? 4D 8D
  constexpr uint64_t oHudMiniMapPosi = 0xB0;     // 16.7 F3 0F 11 93 ? ? ? ? 0F 57 15
  constexpr uint64_t oHudMiniSize = 0xB8;      // 16.7 F3 0F 10 ? ? ? ? ? 48 8B C8 4C 8B E0
 
 
 
  constexpr uint64_t ControlPanel = 0x1D80F60;    // 16.10 48 8B 0D ? ? ? ? BA ? ? ? ? E8 ? ? ? ? 84 C0 75 ? 48 
  constexpr uint64_t IsShowNeuCamp_o1 = 0x8;     // 16.10 48 8B ? ? 80 B9 ? ? ? ? 00 0F 85 ? ? ? ? 48 8B
  constexpr uint64_t IsShowNeuCamp_o2 = 0xE2C;     // 16.10 44 38 A1 ? ? ? ? 0F 84 ? ? ? ? 4C
 
 
 
 
  //Me:
  constexpr uint64_t LocalPlayer = 0x1DFA580;                 // 16.7 48 8B 0D ? ? ? ? 4C ? ? 74 ? 49
 
  //GameObject
  constexpr uint64_t HeroManager = 0x1DC2F40;                 // 16.7  48 ? ? ? ? ? ? 48 ? ? ? ? 33 c0 89 ? ? ? 89 ? ? ? e8 ? ? ? ? 8b  ||48 8B 0D ? ? ? ? 0F 85 ? ? ? ? 83    
  constexpr uint64_t MinionManager = 0x1DC6D48;               // 16.7  48 8B ? ? ? ? ? 44 8B 42 ? 48 C7 42 第一条
  constexpr uint64_t ObjectManager = 0x1DC2EE8;               // 16.7  48 8B 0D ? ? ? ? 8B 10 E8 (all can use) 16.6
  constexpr uint64_t oMgrSizeVec = 0x10;      //   vector型数组
  constexpr uint64_t oMgrSizeList = 0x20;      //   list型数组
  constexpr uint64_t oMgrObj = 0x8;
  //My spell state:
  constexpr uint64_t MySpellState = 0x1DC9AB8;    // 16.7 48 8B 15 ? ? ? ? 85 DB 78 两条都能用  / 84 C0 0F 84 ? ? ? ? 48 8B 1D ? ? ? ? 48 85 DB 0F 84 ? ? ? ? 8B 第三行的mov
  constexpr uint64_t CanNotUse_o1 = 0x24C;     // 16.5  44 89 ? ? ? ? ? 88 ? ? ? ? ? 40
  // constexpr uint64_t CanNotUse_o2 = 8+8*slotID    // 15.16 41 B8 ? ? ? ? 48 8B D6 48 8B 01 FF 90 ? ? ? ? 48 8B 7F //往上数第6行 mov rcx,[rdi+xxx]
  constexpr uint64_t CanNotUse_o3 = 0x26E0;     // 16.7  48 8B 8? ? ? ? ? 48 8B D? E8 ? ? ? ? 48 8D  第一条
 
 
  //Hud instance
  constexpr uint64_t HudInstance = 0x1DC3088;                  // 16.7 48 8B 0D ? ? ? ? 48 85 c9 74 ? 48 8b 49 ? 48 8d  ||   48 8B 05 ? ? ? ? 44 8B 42      
 
  constexpr uint64_t oHudSetup = 0x60;                         // 16.5
  constexpr uint64_t oHudSetupOnlyTargetHero = 0x38;           // 16.5
  constexpr uint64_t oHudSpell = 0x68;                         // 15.20 48 8B 48 ? 48 85 C9 74 ? 48 8B 51 ? 48 85 D2 75  特征2：48 8B 6E ? 0F B6 0D ? ? ? ? 48 85 C0
  constexpr uint64_t oHudMourse = 0x28;                        // 15.16 48 8B 4E ? 0F 28 F0 48 8B 05     OK
  constexpr uint64_t oHudMourseVec3 = 0x34;                    // 15.16 F3 0F 10 4F ? 48 8B BC 24
  constexpr uint64_t o1HudViewCenter = 0x18;      // 15.18 48 8B 49 ? 48 8B 9C 24 ? ? ? ? 48 83 C4
  constexpr uint64_t o2HudViewCenter = 0x1A0;      // 15.23 8B 82 ? ? ? ? 41 89 00 8B 82 第一条
  constexpr uint64_t Zoom = 0x1D80F68;       // 15.14 48 83 3D ? ? ? ? 00 0F 84 ? ? ? ? 48 8B 0D ? ? ? ? 48 8B 49 ? E8 ? ? ? ? 0F cmp里面的 15.9  :E8 ? ? ? ? F3 0F 10 78 ? F3 0F 5E FE 进入这个call，里面的源操作数
  constexpr uint64_t oZoom = 0x28;
  constexpr uint64_t ZoomCoefficient = 0x1DC3088;     // 16.6 48 8B 0D ? ? ? ? 48 85 c9 74 ? 48 8b 49 ? 48 8d //第一条
  constexpr uint64_t ZoomCoefficiento1 = 0x18;     // 往下数第3条 48 8B 41 ? F3 0F 10 88 ? ? ? ? F3 0F 11 4C ? ?
  constexpr uint64_t ZoomCoefficiento2 = 0x318;     // 16.5 F3 44 0F 59 8F ? ? ? ? 45
 
 
  //
  constexpr uint64_t TerSafeVtable = 0x1DC2ED0;    // 16.7  48 8B 1D ? ? ? ? 48 85 DB 0F 84 ? ? ? ? 48 8D 05 ? ? ? ? 48 89 BD 随便一条
  constexpr uint64_t TSVo0 = 0x638;       // 15.17 48 8B BE ? ? ? ? 48 85 FF 74 ? 66
  constexpr uint64_t TSVo1 = 0x10;       // 15.16 49 8B 4D ? 89 44 24 ? 48 8D 45 ?
  constexpr uint64_t TSVo2 = 0x0;        // 15.16 48 8B ? FF 50 ? 85 C0 0F 85 ? ? ? ? 48 8D
  constexpr uint64_t TSVo3 = 0x10;       // 15.16 FF 50 ? 85 C0 0F 85 ? ? ? ? 48 8D
 
  //Under Mouse Object
  constexpr uint64_t UnderMouseObj = 0x1DC6F60;               // 16.7 48 89 0D ? ? ? ? 48 8D 05 ? ? ? ? 48 89 01 33 D2  第一条     OK
  constexpr uint64_t oUnderMouseObj = 0x18;     // 15.20 48 8B 4B ? 0F 28 74 24 ? 8B 81
 
 
  constexpr uint64_t FirstACEhook = 0x100000000;    // 15.16 E8 ? ? ? ? 41 54 41 55
  constexpr uint64_t LastACEhook = 0x10005E000;    //0x100054000;    // 16.6 59 5B可疑钩子
  constexpr uint64_t ACEhoosOffset = 0x1000;     // 15.16
  constexpr uint64_t NopOffset = 0x75;      // 15.16
  constexpr uint64_t NopSize = 0x2;       // 15.16
  //League of Legends.exe+9379D0 - E8 2B366DBF           - call 10000B000
 
 
 }
 
 namespace BasicAttack
 {
  constexpr uint64_t BA_base = 0x2C68;      //48 8B 81 ? ? ? ? C3 CC 40 56 48
  constexpr uint64_t BA_o1 = 0x2C0;       //5B C3 CC 48 8B 81 ? ? ? ? C3
  constexpr uint64_t BA_o2 = 0x38;       //48 8B 41 ? 48 89 02 4C 89 42 08 F0 41倒数第二条
 
  //远程：
  constexpr uint64_t BA_remote = 0x8 + 0x145;
  //远程：145  41 C6 80 ? ? ? ? 01 49 8B
  // 近战 143 ：League of Legends.exe+90B4E7 - 40 88 AF 43010000     - mov [rdi+00000143],bpl
 
 }
 
 namespace Func
 {
 
 
  constexpr uint64_t CastSpell2NoAceHook = 0x8FDD27;   // 15.20.1  48 89 48 ? 55 56 57 41 54 41 55 48
  constexpr uint64_t CastSpell2CheckFlag = 0x1D79CF0;   // 15.18  88 44 24 ? 48 FF E1 C3 CC 往上数第7个mov r?x xxx
  // Object Type:
  constexpr uint64_t IsAlive = 0x2D8CC0;      // 15.22  E8 ? ? ? ? 84 C0 74 ? 48 8B 83 ? ? ? ? 48 8D 8B
  constexpr uint64_t IsHero = 0x2FA4E0;      // 15.22  E8 ? ? ? ? 84 C0 0F 85 ? ? ? ? 48 8B CB E8 ? ? ? ? 84 C0 74 ? 48 8B   15.3
 
  //Object Manager:
  constexpr uint64_t GetFirstObject = 0x526FA0;    // 16.7  48 83 EC ? 48 8B 51 ? 8B 41 ? 48 8D 0C C2 注意要再往上返回一级，否则调用前RCX+0x18(我的代码里已经+0x18了) ||
  constexpr uint64_t GetNextObject = 0x5279E0;    // 16.7  0F B7 42 ? 44 8B 41 所在的函数 || E8 ? ? ? ? 48 8B D8 48 85 C0 0F 85 ? ? ? ? 0F 28 74 24 ? 48 8B B4 24
 
  constexpr uint64_t CompareObjectTypeFlags = 0x2A05A0;  // 16.2  40 56 48 83 EC ? 0F B6 41
 
  constexpr uint64_t GetAttackDelay = 0x53DBA0;    // 16.7  F3 0F 10 89 ? ? ? ? E9 ? ? ? ? 所在的函数 E8 ? ? ? ? 33 C0 F3 0F 11 83
  constexpr uint64_t GetAttackCastDelay = 0x53DAA0;   // 16.7  48 89 5C 24 ?? 48 89 74 24 ?? 57 48 83 EC 60 48 8B 01 8B DA 0F 29 74 24 ??
 
  constexpr uint64_t GetBoundingRadius = 0x27AFB0;   // 15.22  40 53 48 83 EC ? 48 83 B9 ? ? ? ? 00 48 8B D9 0F 29 74 24 20 
  constexpr uint64_t GetAiManager = 0x2840E0;     // 15.22  73 20 90 0F B6 0C 02 所在的函数
 
  constexpr uint64_t oSpellBook = 0x30E8;      // 16.6  49 8D ? ? ? ? ? 8B D0 4C 8D ? ? E8
  constexpr uint64_t oSpellSlot = 0xAE0;      // 16.6  48 63 C2 48 8B 84 C1 下一行的值
  constexpr uint64_t oSpellInfo = 0x128;      // 15.17  48 83 BA ? ? ? ? 00 74 03 B0 01
  constexpr uint64_t oSpellInfoInput = 0x120;     // 15.11
  constexpr uint64_t oSpellMeVecInput = 0x18;     // 15.11
  constexpr uint64_t oSpellStartVecInput1 = oSpellMeVecInput + 0xC;   // 15.11
  constexpr uint64_t oSpellStartVecInput2 = oSpellStartVecInput1 + 0xC;  // 15.11
  constexpr uint64_t oSpellEndVecInput = oSpellStartVecInput2 + 0xC;   // 15.11
  constexpr uint64_t oSpellTargetVecInput = 0x24;        // 15.11
 
 
  constexpr uint64_t GetPing = 0x674BB0;      // 16.7 E8 ? ? ? ? 8B F8 39 03 OK
 
 
  // System Chat:
  constexpr uint64_t PrintChatSys = 0xA521F0;     // 15.12  E8 ? ? ? ? 4C 8B C3 B2 01
  constexpr uint64_t PrintChatSysRcx = 0x1C17288;    // 15.12  48 8B 0D ? ? ? ? E8 ? ? ? ? 48 8B B4 24 ? ? ? ? EB
  constexpr uint64_t PrintChatSysR8d = 0x183C738;    // 15.1?  Change too large, maybe incorrect. 44 8B 05 ? ? ? ? 48 8B 54 24 ? 48 8B 0D ? ? ? ? E8 ? ? ? ? 48 8B B4
 
  constexpr uint64_t PrintChatPublic = 0x6964D0;    // 16.6   41 FF C0 EB ? 41 B8 下面那个call xxx 
  constexpr uint64_t PrintChatPublicInstance = 0x1DA1440;  // 16.6   41 FF C0 EB ? 41 B8 下面那个mov rcx,xxx
 
 
 
  // terrain and coordiante:
  constexpr uint64_t WorldToScreen = 0x12812D0;    //16.7  E8 ? ? ? ? F3 0F 10 44 24 ? F3 41 0F 11 06 15.6
  constexpr uint64_t ViewPort = 0x1DC6D00;     //16.7  48 8B 15 ? ? ? ? 48 8B 0D ? ? ? ? 48 81 第一条      15.6
  constexpr uint64_t oViewPort = 0x2B0;      //16.6  48 8B AE ? ? ? ? 0F 11 
 
 
 
  constexpr uint64_t OnCreateObject = 0x532120;     //16.10  48 83 C4 ? 5E 48 FF A0 所在的函数 17个字节
  constexpr uint64_t OnDeleteObject = 0x5223C0;     //16.10  E8 ? ? ? ? 48 83 C3 08 48 3B DE 75 DE 18个字节
 
 
  //ping singal ::100037000  6781D4
 
 }

}

namespace Offset
{
 namespace oGameObject
 {
  // GameObject:
  //constexpr uint64_t oObjLiveState = 0x43;
  constexpr uint64_t oObjNetID = 0xC4;      //15.17  8B 80 ? ? ? ? 89 43 ? 48 83 C4 ? 5B C3 8B  //放指向性技能用
  constexpr uint64_t oObjLiveState = 0x258;     //16.2
  constexpr uint64_t oObjTeamID = 0x259;      //16.3  0F B6 88 ? ? ? ? EB 07 0F B6 8E
  constexpr uint64_t oObjPosition = 0x25C;     //16.2  F3 0F 10 B6 ? ? ? ? 0F 29 7C 24 
  constexpr uint64_t oObjEffectData = 0x260;     //15.11
  constexpr uint64_t oObjIsVisible = 0x308;     //16.2  两个偏移加起来： 48 8D 8F ? ? ? ? 33 D2 4C 8B(invalid) / 80 79 ? 00 75 1B 32 C0
  constexpr uint64_t oObjMana = 0x340;      //invalid
  constexpr uint64_t oObjMaxMana = 0x368;      //invalid
  constexpr uint64_t oObjState = 0x5A8;      //是否处于“正在引导”的状态，int，0x400
  constexpr uint64_t oObjIsTargetable = 0xED0;    // 16.3  0F B6 83 ? ? ? ? 48 83 C4 ? 5B , first item
  constexpr uint64_t oObjRecallState = 0x1;     //invalid
  constexpr uint64_t oObjHealth = 0x1080;      // 16.3 xxxxxx
  constexpr uint64_t oObjMaxHealth = 0x10A8;     //invalid
  constexpr uint64_t oObjActionState = 0x1;     //invalid
  constexpr uint64_t oObjBonusAttackDamage = 0x1CB8;   //invalid
  constexpr uint64_t oObjAbilityPower = 0x1718;    // 15.12 法术强度！！！ 必须查找访问
  constexpr uint64_t oObjBaseAttackDamage = 0x1EC8;   //invalid
  constexpr uint64_t oObjScale = 0x1;       //invalid
  constexpr uint64_t oObjArmor = 0x1804;      // 15.11 
  constexpr uint64_t oObjMagicRes = 0x17DC;     // 16.3  8B 8B ? ? 00 00 89 ? ? 8B 8B ? ? 00 00 89 第二条是护甲，第一条是魔抗
  constexpr uint64_t oObjMagicResPenFlat = 0x2308;   //15.12
  constexpr uint64_t oObjMagicResPenMod = 0x2358;    //15.12
  constexpr uint64_t oObjArmorPenFlat = 0x1;     //invalid
  constexpr uint64_t oObjArmorPenMod = 0x1;     //invalid
  constexpr uint64_t oObjMoveVelosity = 0x17DC;    //Using AiManager's Speed is better(include dash speed)
  constexpr uint64_t oObjAttackRange = 0x17F4;    //15.23  F3 0F 10 80 ? ? ? ? F3 0F 11 47
  constexpr uint64_t oObjBuffManager = 0x2E70;    //16.xxx   GameObject + 0x2E30 -> BuffManager* // 48 8D 8B ? ? ? ? 48 8B D7 E8 ? ? ? ? ? ? ? 48 8B CB
  constexpr uint64_t oObjSpellBook = 0x30E8;     //16.3  49 8D ? ? ? ? ? 8B D0 4C 8D ? ? E8
  constexpr uint64_t oObjOnCastingSpell = oObjSpellBook + 0x38;
  constexpr uint64_t oObjSpellSlot = 0xAE0;     //15.23  48 8D ? ? ? ? ? ? 76 ? B8 ? ? ? ? 48 随便一条
  constexpr uint64_t oObjMonsterName = 0x136C; //0x1354;  //15.17  
  constexpr uint64_t oObjCharacterData = 0x4030;    //16.6  48 8B 8B ? ? ? ? 33 ED 48
  constexpr uint64_t oCharacterDataHash = 0x10;    //16.6 自己测试
  constexpr uint64_t oCharacterDataData = 0x28;
  constexpr uint64_t oObjName = 0x4328;      // 16.7 
  constexpr uint64_t oObjLevel = 0x4CF0;      //invalid
  constexpr uint64_t oObjAiManager = 0x41A8;     //UnKnowCheat user:giaanthunder
  constexpr uint64_t oObjDragonName = 0x1348;     //invalid
  constexpr uint64_t oObjBasicAttackCastCount = 0x59A8;  //CC 48 83 EC ? 4C 8D 81 ? ? ? ? 4C 8B D9 第三行和第第五行加起来

 }
 
 
 namespace EffectEmitter
 {
  constexpr uint64_t Data = 0x260;
  constexpr uint64_t Name = 0x60;        // 15.12_
  constexpr uint64_t Caster = 0x40;
  constexpr uint64_t FirstCaster = 0x8;
  constexpr uint64_t Target = 0x30;
  constexpr uint64_t FirstTarget = 0x8;
 }
 
 
 namespace AiManager
 {
  constexpr uint64_t oObjAiMgrTargetPos = 0x34;       // 15.11
  constexpr uint64_t oObjAiMgrVelocity = 0x318;       // 15.11
  constexpr uint64_t oObjAiMgrIsMoving = 0x31C;       // 15.11
  constexpr uint64_t oObjAiMgrCurrentSegment = 0x320;      // 15.11
  constexpr uint64_t oObjAiMgrStartPath = 0x330;       // 15.11
  constexpr uint64_t oObjAiMgrTargetPosition = oObjAiMgrStartPath + 0xC;   // 15.11
  constexpr uint64_t oObjAiMgrNavArray = 0x348;       // 15.11
  constexpr uint64_t oObjAiMgrSegmentsCount = 0x350;      // 15.11
  constexpr uint64_t oObjAiMgrDashSpeed = 0x360;       // 15.11
  constexpr uint64_t oObjAiMgrIsDashing = 0x384;       // 15.11 
  //constexpr uint64_t oObjAiMgrTargetPosition = 0x3A8;      // 15.11 0x33C 这个貌似更基本
  constexpr uint64_t oObjAiMgrMoveVec3 = oObjAiMgrStartPath + 0x150;  // 15.11 0x480
  constexpr uint64_t oObjAiMgrServerPos = 0x474;       // 15.11
 
 };
 
 namespace SpellDataScript
 {
  constexpr uint64_t Name = 0x8;
  constexpr uint64_t Hash = 0x18;
 }
 
 namespace SpellData
 {
  constexpr uint64_t SpellDataScript = 0x18;
  constexpr uint64_t SpellName = 0x28;
  constexpr uint64_t SpellDataResource = 0x60;
 }
 namespace oSpellSlot
 {
  constexpr uint64_t oSpellLevel = 0x28;      // 15.11
  constexpr uint64_t oNextReadyTime = 0x30;      // 15.11 技能下次准备就绪的时间 F3 0F 11 43 ? 48 8B 46 15.6
  constexpr uint64_t oSpellStartTime = 0x34;     // 15.11 本次技能开始的时间，例如泽拉斯的R，引导结束后会清零 C7 45 ?? 00 00 00 00 EB 07 15.6
  constexpr uint64_t oSpellCoolTime = 0x74;     // 15.11 上次释放的技能的冷却时间
 }
 
 namespace oOnCastingSpellData
 {
  constexpr uint64_t SpellInfo = 0x8;
  constexpr uint64_t CastSpellName = 0x28;
  constexpr uint64_t oStartPosition = 0xD0;
 
  constexpr uint64_t oTargetPosition = 0xDC; //oStartPosition + 0xC
 }
 
 namespace SpellInfo
 {
  constexpr uint64_t SpellData = 0x0;
 
  constexpr uint64_t SrcIndex = 0x88;
 
  constexpr uint64_t StartPos = 0xA4;
  constexpr uint64_t EndPos = StartPos + 0xC;
  constexpr uint64_t CastPos = EndPos + 0xC;
 
  constexpr uint64_t TargetIndex = 0xE0;
  constexpr uint64_t CastDelay = 0xF0;
 
  constexpr uint64_t IsSpell = 0x10C;  // == 0
  constexpr uint64_t IsSpecialAttack = 0x112;// == 0
  constexpr uint64_t IsAuto = 0x113;
 
  constexpr uint64_t Slot = 0x11C;
 };
 
 
 namespace MissileData
 {
  constexpr uint64_t SpellInfo = 0x0260;
  constexpr uint64_t SrcIdx = 0x2C4; // 0x2DC;
  constexpr uint64_t DestCheck = 0x31C;
  constexpr uint64_t DestIdx = 0x318; // 0x330;
  constexpr uint64_t StartPos = 0x2E0;
  constexpr uint64_t CurPos = 0x1DC;
  constexpr uint64_t EndPos = 0x2EC;
 }
 
 
 
 namespace BuffData
 {
  // --- Buff Manager Offsets ---
  constexpr uint64_t oBuffManagerArray = 0x18; // BuffManager + 0x18 -> start 
  constexpr uint64_t oBuffManagerArrayEnd = 0x20; // BuffManager + 0x20 -> end
  constexpr uint64_t oBuffInstanceBuffScript = 0x10; // BuffInstance + 0x10 -> BuffScript*
  constexpr uint64_t oBuffScriptBuffName = 0x8; // BuffScript + 0x8 -> RiotString
  constexpr uint64_t oBuffInstanceType = 0xc; // BuffInstance + 0x8 -> int (buff type)
  constexpr uint64_t oBuffInstanceStartTime = 0x18; // BuffInstance + 0x18 -> float (start time)
  constexpr uint64_t oBuffInstanceEndTime = 0x1C; // BuffInstance + 0x1C -> float (end time)
  constexpr uint64_t oBuffInstanceDuration = 0x20; // BuffInstance + 0x20 -> float (duration)
  constexpr uint64_t oBuffInstanceStackCount = 0x38; // BuffInstance + 0x38 -> int (stack count)
  constexpr uint64_t oBuffInstanceCount = 0x78; // BuffInstance + 0x78 -> int (count)
 }
}
__________________
HackShield
sq834960394 is offline

Old 7th April 2026, 08:50 PM   #13011
ibrahimcelik
Junior Member

ibrahimcelik's Avatar

Join Date: Jul 2023
Posts: 51
Reputation: 10
Rep Power: 67
ibrahimcelik has made posts that are generally average in quality
Points: 2,241, Level: 4
Points: 2,241, Level: 4 Points: 2,241, Level: 4 Points: 2,241, Level: 4
Level up: 21%, 559 Points needed
Level up: 21% Level up: 21% Level up: 21%
Activity: 99.0%
Activity: 99.0% Activity: 99.0% Activity: 99.0%
Last Achievements
League of Legends Reversal, Structs and Offsets
@kyudev no, IsDead = 0x1D7A6340;

anyone know DrawCircle pattern?

anyone know DrawCircle pattern?
ibrahimcelik is online now

Old Yesterday, 12:42 AM   #13012
sq834960394
h4x0!2

sq834960394's Avatar

Join Date: Dec 2023
Location: Tokyo
Posts: 92
Reputation: 627
Rep Power: 58
sq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiendsq834960394 is the Coding Fiend
Points: 2,765, Level: 4
Points: 2,765, Level: 4 Points: 2,765, Level: 4 Points: 2,765, Level: 4
Level up: 95%, 35 Points needed
Level up: 95% Level up: 95% Level up: 95%
Activity: 6.3%
Activity: 6.3% Activity: 6.3% Activity: 6.3%
Last Achievements
League of Legends Reversal, Structs and Offsets
Quote:
Originally Posted by ibrahimcelik View Post
@kyudev no, IsDead = 0x1D7A6340;

anyone know DrawCircle pattern?

anyone know DrawCircle pattern?
i have search it for a long time, but failed...
__________________
HackShield
sq834960394 is offline

Reply Submit Thread to Reddit RedditSubmit Thread to Twitter TwitterSubmit Thread to Facebook Facebook 
Page 651 of 651 « First < 151 551 601 641 647 648 649 650 651 

Tags
typedef, #define, offsets, pobj;, int, float, updated, bool, thread, dword

« Previous Thread | Next Thread »

Forum Jump

    League of Legends

All times are GMT. The time now is 01:45 AM.
Copyright ©2000-2026, Unknowncheats™
DMCA - Contact
Terms of Use - Privacy Policy - Forum Rules
